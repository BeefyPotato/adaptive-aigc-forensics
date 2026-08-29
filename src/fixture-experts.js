function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function fixtureRgbExpert(rgb) {
  const brightness = mean(rgb.map(([red, green, blue]) => (red + green + blue) / 3));
  return (brightness - 0.5) * 4;
}

export function fixtureSignalExpert({ luminance, width, height }) {
  const differences = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = row * width + column;
      if (column + 1 < width) {
        differences.push(Math.abs(luminance[index] - luminance[index + 1]));
      }
      if (row + 1 < height) {
        differences.push(Math.abs(luminance[index] - luminance[index + width]));
      }
    }
  }
  return (mean(differences) - 0.15) * 8;
}

function sigmoid(logit) {
  if (logit >= 0) {
    return 1 / (1 + Math.exp(-logit));
  }
  const exponential = Math.exp(logit);
  return exponential / (1 + exponential);
}

export function equalLogitFusion(rgbLogit, signalLogit) {
  return sigmoid((rgbLogit + signalLogit) / 2);
}
