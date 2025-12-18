/**
 * Hash a frame to detect duplicates
 * Uses a simple hash of image data to detect identical frames
 */
export function hashFrame(imageData: ImageData): string {
  // Sample pixels from the image to create a hash
  // This is faster than hashing the entire image
  const data = imageData.data;
  const sampleSize = Math.min(100, data.length / 4); // Sample 100 pixels
  const step = Math.floor(data.length / 4 / sampleSize);
  
  let hash = 0;
  for (let i = 0; i < sampleSize; i++) {
    const idx = i * step * 4;
    if (idx < data.length - 3) {
      // Use RGB values for hashing
      hash = ((hash << 5) - hash) + data[idx];     // R
      hash = ((hash << 5) - hash) + data[idx + 1]; // G
      hash = ((hash << 5) - hash) + data[idx + 2]; // B
      hash = hash & hash; // Convert to 32-bit integer
    }
  }
  
  return hash.toString(36); // Convert to base36 string
}

/**
 * Check if a frame should be sent (not a duplicate)
 */
export function shouldSendFrame(
  frameHash: string,
  lastFrameHash: string | null
): boolean {
  if (lastFrameHash === null) {
    return true; // First frame always send
  }
  return frameHash !== lastFrameHash;
}



