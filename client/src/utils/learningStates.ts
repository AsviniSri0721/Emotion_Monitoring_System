/**
 * Learning states derived from concentration and emotion trends
 * These represent the student's engagement level in an educational context
 */
export type LearningState =
  | "ENGAGED"
  | "PASSIVE"
  | "STRUGGLING"
  | "DISENGAGED"
  | "FATIGUED";

/**
 * Learning states that should trigger interventions
 */
export const INTERVENTION_STATES: LearningState[] = ["DISENGAGED", "FATIGUED"];

/**
 * Required number of consecutive frames a state must persist before action
 */
export const REQUIRED_PERSISTENCE_FRAMES = 15;

/**
 * Derive learning state from concentration score and emotion probabilities
 * 
 * @param concentration - Concentration score (0-100)
 * @param emotions - Object with emotion probabilities
 * @returns LearningState classification
 */
export function deriveLearningState(
  concentration: number,
  emotions: Record<string, number>
): LearningState {
  // High concentration = engaged
  if (concentration >= 70) {
    return "ENGAGED";
  }

  // Moderate concentration = passive (normal learning state)
  if (concentration >= 40 && concentration < 70) {
    return "PASSIVE";
  }

  // Low concentration with confusion/frustration = struggling
  if (
    concentration < 40 &&
    ((emotions.confusion || 0) + (emotions.frustration || 0)) > 0.3
  ) {
    return "STRUGGLING";
  }

  // Low concentration with boredom = disengaged
  // BUT: Require higher boredom threshold and lower concentration to avoid false positives
  // This prevents single-frame boredom spikes from triggering disengagement
  if (
    concentration < 40 &&
    (emotions.boredom || 0) > 0.5  // Increased threshold from 0.2 to 0.5
  ) {
    return "DISENGAGED";
  }

  // Very low concentration = fatigued
  if (concentration < 30) {
    return "FATIGUED";
  }

  // Default to passive for edge cases
  return "PASSIVE";
}

/**
 * Get supportive intervention message based on learning state
 * Uses educational, non-judgmental language
 * 
 * @param state - The learning state that triggered the intervention
 * @returns Supportive message text
 */
export function getInterventionMessage(state: LearningState): string {
  switch (state) {
    case "DISENGAGED":
      return "It looks like you may benefit from a short break. Would you like to try a quick interactive activity?";
    case "FATIGUED":
      return "You might benefit from a brief pause. Would you like a short recap or example to help refocus?";
    default:
      return "Would you like a quick recap or example?";
  }
}

/**
 * Get intervention title based on learning state
 * 
 * @param state - The learning state that triggered the intervention
 * @returns Supportive title text
 */
export function getInterventionTitle(state: LearningState): string {
  switch (state) {
    case "DISENGAGED":
      return "Take a Moment";
    case "FATIGUED":
      return "Quick Break Suggested";
    default:
      return "Learning Support";
  }
}

