export const calculateDrawdown = (cumulativeReturns) => {
  let peak = -Infinity;
  const drawdowns = cumulativeReturns.map(val => {
    if (val > peak) peak = val;
    const dd = peak === 0 ? 0 : (val - peak) / peak;
    return dd;
  });
  return drawdowns;
};

export const calculateCAGR = (startValue, endValue, years) => {
  if (startValue <= 0 || years <= 0) return 0;
  return Math.pow(endValue / startValue, 1 / years) - 1;
};

export const calculateSharpe = (returns, riskFreeRate = 0.0) => {
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  return stdDev === 0 ? 0 : ((mean - riskFreeRate) / stdDev) * Math.sqrt(12);
};
