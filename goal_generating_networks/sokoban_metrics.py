def compute_sokoban_metrics(pred_board, input_board, target_board, goal_board):
    """
    Computes a comprehensive set of Sokoban-specific metrics.
    
    All inputs: (B, 144) torch.long tensors
    
    Token IDs:
        0: Floor, 1: Wall, 2: Goal, 3: Box,
        4: BoxOnGoal, 5: Agent, 6: AgentOnGoal, 7: MASK
    """
    metrics = {}
    B = pred_board.size(0)

    # ─────────────────────────────────────────────────
    # LEVEL 1: TOKEN DISTRIBUTION METRICS
    # ─────────────────────────────────────────────────

    token_names = {0: 'floor', 1: 'wall', 2: 'goal', 3: 'box',
                   4: 'box_on_goal', 5: 'agent', 6: 'agent_on_goal'}

    for token_id, name in token_names.items():
        pred_count  = (pred_board   == token_id).float().sum(dim=1)  # (B,)
        target_count = (target_board == token_id).float().sum(dim=1) # (B,)

        # Mean absolute error in count per sample
        metrics[f'count_mae/{name}'] = (pred_count - target_count).abs().mean().item()

        # Fraction of samples where count is exactly correct
        metrics[f'count_exact/{name}'] = (pred_count == target_count).float().mean().item()

    # ─────────────────────────────────────────────────
    # LEVEL 2: STRUCTURAL VALIDITY METRICS
    # ─────────────────────────────────────────────────

    # --- 2a. Agent count validity ---
    # A valid board has exactly 1 agent (token 5 or 6)
    agent_count = (
        (pred_board == 5).float() + 
        (pred_board == 6).float()
    ).sum(dim=1)  # (B,)
    
    metrics['validity/agent_count_exact'] = (agent_count == 1).float().mean().item()
    metrics['validity/agent_count_mae']   = (agent_count - 1).abs().float().mean().item()

    # --- 2b. Box conservation ---
    # Total boxes (3 + 4) in pred must equal total boxes in target
    pred_total_boxes   = ((pred_board == 3) | (pred_board == 4)).float().sum(dim=1)
    target_total_boxes = ((target_board == 3) | (target_board == 4)).float().sum(dim=1)
    input_total_boxes  = ((input_board == 3) | (input_board == 4)).float().sum(dim=1)

    metrics['validity/box_count_exact']   = (pred_total_boxes == target_total_boxes).float().mean().item()
    metrics['validity/box_count_mae']     = (pred_total_boxes - target_total_boxes).abs().mean().item()

    # --- 2c. Goal conservation ---
    # Total goals (2 + 4 + 6) must be preserved from the input board
    # Goals are static — they never move
    def count_goals(board):
        return ((board == 2) | (board == 4) | (board == 6)).float().sum(dim=1)

    pred_total_goals  = count_goals(pred_board)
    input_total_goals = count_goals(input_board)

    metrics['validity/goal_count_exact'] = (pred_total_goals == input_total_goals).float().mean().item()
    metrics['validity/goal_count_mae']   = (pred_total_goals - input_total_goals).abs().mean().item()

    # --- 2d. Box-Goal balance ---
    # num_boxes + num_boxes_on_goal must equal num_goals + num_boxes_on_goal
    # which simplifies to: total_boxes == total_goals
    pred_goals_count = count_goals(pred_board)
    box_goal_balanced = (pred_total_boxes == pred_goals_count)
    metrics['validity/box_goal_balanced'] = box_goal_balanced.float().mean().item()

    # --- 2e. Wall preservation ---
    # Walls in input_board must remain walls in pred_board
    input_walls = (input_board == 1)
    wall_preserved = (pred_board == 1) & input_walls
    wall_accuracy = wall_preserved.float().sum(dim=1) / (input_walls.float().sum(dim=1) + 1e-6)
    metrics['validity/wall_preservation'] = wall_accuracy.mean().item()

    # Penalty: non-wall tokens predicted as wall (hallucinated walls)
    hallucinated_walls = (pred_board == 1) & (~input_walls)
    metrics['validity/hallucinated_walls_per_sample'] = hallucinated_walls.float().sum(dim=1).mean().item()

    # --- 2f. Fully valid boards ---
    # A board passes ALL structural checks
    fully_valid = (
        (agent_count == 1) &
        (pred_total_boxes == target_total_boxes) &
        (pred_total_goals == input_total_goals) &
        (wall_accuracy > 0.99)
    )
    metrics['validity/fully_valid_board'] = fully_valid.float().mean().item()

    # ─────────────────────────────────────────────────
    # LEVEL 3: SUBGOAL QUALITY METRICS
    # ─────────────────────────────────────────────────

    # --- 3a. Boxes on goal: are we making progress? ---
    # Compare number of boxes-on-goal in: input → pred → goal_board
    input_boxes_on_goal  = (input_board  == 4).float().sum(dim=1)
    pred_boxes_on_goal   = (pred_board   == 4).float().sum(dim=1)
    target_boxes_on_goal = (target_board == 4).float().sum(dim=1)
    goal_boxes_on_goal   = (goal_board   == 4).float().sum(dim=1)

    # Did we improve on the input? (Positive = progress made)
    progress_delta = pred_boxes_on_goal - input_boxes_on_goal
    metrics['subgoal/boxes_on_goal_delta']    = progress_delta.mean().item()
    metrics['subgoal/boxes_on_goal_improved'] = (progress_delta > 0).float().mean().item()
    metrics['subgoal/boxes_on_goal_regressed']= (progress_delta < 0).float().mean().item()

    # How close is pred to the fully solved board in terms of boxes on goal?
    max_boxes  = goal_boxes_on_goal  # The solved board has all boxes on goals
    pred_gap   = (max_boxes - pred_boxes_on_goal).clamp(min=0)
    target_gap = (max_boxes - target_boxes_on_goal).clamp(min=0)
    metrics['subgoal/boxes_solved_gap_pred']   = pred_gap.mean().item()
    metrics['subgoal/boxes_solved_gap_target'] = target_gap.mean().item()

    # --- 3b. Directional progress score ---
    # Is pred_board strictly between input and goal?
    # Measured by: does pred have more boxes-on-goal than input
    #              AND fewer than or equal to goal?
    directional = (pred_boxes_on_goal >= input_boxes_on_goal) & \
                  (pred_boxes_on_goal <= goal_boxes_on_goal)
    metrics['subgoal/directional_progress'] = directional.float().mean().item()

    # --- 3c. Spatial box displacement ---
    # How far did boxes move compared to the target displacement?
    # Reshape to 2D for position analysis
    pred_2d   = pred_board.view(B, 12, 12)
    input_2d  = input_board.view(B, 12, 12)
    target_2d = target_board.view(B, 12, 12)

    # Binary masks of where boxes are (including BoxOnGoal)
    pred_box_map   = ((pred_2d   == 3) | (pred_2d   == 4)).float()
    input_box_map  = ((input_2d  == 3) | (input_2d  == 4)).float()
    target_box_map = ((target_2d == 3) | (target_2d == 4)).float()

    # L1 distance between box heatmaps (rough measure of box layout similarity)
    pred_vs_target_box  = (pred_box_map  - target_box_map).abs().sum(dim=(1,2))
    input_vs_target_box = (input_box_map - target_box_map).abs().sum(dim=(1,2))

    metrics['subgoal/box_layout_l1_pred_vs_target']  = pred_vs_target_box.mean().item()
    metrics['subgoal/box_layout_l1_input_vs_target'] = input_vs_target_box.mean().item()

    # Improvement ratio: did pred get closer to target than input was?
    # > 0.5 means on average the model reduces the layout distance
    improvement = (pred_vs_target_box < input_vs_target_box).float()
    metrics['subgoal/box_layout_improved'] = improvement.mean().item()

    return metrics