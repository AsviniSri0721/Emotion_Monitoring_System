// Emotion weights for concentration calculation
export const EMOTION_WEIGHTS: Record<string, number> = {
  focus: 1.0,
  happy: 0.8,
  neutral: 0.5,
  surprise: 0.6,
  confusion: 0.3,
  frustration: 0.2,
  boredom: 0.1,
  sleepy: 0.0,
};

/**
 * Calculate concentration score from emotion probabilities
 * @param emotions - Object with emotion names as keys and probabilities as values
 * @returns Concentration score (0-100)
 */
export function calculateConcentration(emotions: Record<string, number>): number {
  let score = 0;
  for (const [emotion, prob] of Object.entries(emotions)) {
    score += (EMOTION_WEIGHTS[emotion] ?? 0) * prob;
  }
  return Math.round(score * 100);
}

/**
 * Get the dominant emotion (emotion with highest probability)
 * @param emotions - Object with emotion names as keys and probabilities as values
 * @returns Emotion name with highest probability
 */
export function getMaxEmotion(emotions: Record<string, number>): string {
  if (!emotions || Object.keys(emotions).length === 0) {
    return 'neutral';
  }
  return Object.entries(emotions).reduce((max, [emotion, prob]) => 
    prob > max[1] ? [emotion, prob] : max
  )[0];
}

/**
 * All 8 emotions that should always be displayed
 */
export const ALL_EMOTIONS = ['focus', 'happy', 'neutral', 'surprise', 'confusion', 'frustration', 'boredom', 'sleepy'];

/**
 * Average emotion probabilities across multiple frames
 * Ensures all 8 emotions are present (set to 0 if missing)
 * @param emotionBuffer - Array of emotion probability objects
 * @returns Averaged emotion probabilities with all 8 emotions
 */
export function averageEmotions(emotionBuffer: Record<string, number>[]): Record<string, number> {
  if (emotionBuffer.length === 0) {
    // Return all emotions with 0 if buffer is empty
    const empty: Record<string, number> = {};
    ALL_EMOTIONS.forEach(emotion => empty[emotion] = 0);
    return empty;
  }

  const allEmotions = new Set<string>(ALL_EMOTIONS);
  emotionBuffer.forEach(emotions => {
    Object.keys(emotions).forEach(emotion => allEmotions.add(emotion));
  });

  const averaged: Record<string, number> = {};
  for (const emotion of allEmotions) {
    const sum = emotionBuffer.reduce((acc, emotions) => acc + (emotions[emotion] || 0), 0);
    averaged[emotion] = sum / emotionBuffer.length;
  }

  // Ensure all 8 emotions are present (set to 0 if missing)
  ALL_EMOTIONS.forEach(emotion => {
    if (!(emotion in averaged)) {
      averaged[emotion] = 0;
    }
  });

  return averaged;
}

/**
 * Smoothing window size for temporal smoothing
 */
export const SMOOTHING_WINDOW = 15;

