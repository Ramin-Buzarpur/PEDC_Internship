export const generateMockData = () => {
  const data = [];
  let currentValues = {
    'Passive (S&P500)': 1.0,
    'AR1x': 1.0,
    'LSTM': 1.0,
    'xLSTM': 1.0,
    'LPatchTST': 1.0,
    'VLSTM': 1.0
  };

  const modelsConfig = {
    'Passive (S&P500)': { mean: 0.004, vol: 0.045 },
    'AR1x': { mean: 0.006, vol: 0.04 },
    'LSTM': { mean: 0.010, vol: 0.035 },
    'xLSTM': { mean: 0.014, vol: 0.03 },
    'LPatchTST': { mean: 0.018, vol: 0.028 },
    'VLSTM': { mean: 0.019, vol: 0.025 } 
  };

  const years = 15;
  const months = years * 12;
  const startDate = new Date(2010, 0, 1);

  for (let i = 0; i <= months; i++) {
    const currentDate = new Date(startDate.getFullYear(), startDate.getMonth() + i, 1);
    const dataPoint = {
      date: currentDate.toISOString().split('T')[0],
      displayDate: currentDate.toLocaleDateString('en-US', { year: 'numeric', month: 'short' }),
      year: currentDate.getFullYear(),
      rawReturns: {}
    };

    const marketShock = (Math.random() - 0.5) * 0.1;
    const isCrisis = i === 120; // Simulate 2020 crash

    Object.keys(modelsConfig).forEach(model => {
      if (i === 0) {
        dataPoint[model] = 1.0;
        dataPoint.rawReturns[model] = 0;
      } else {
        const { mean, vol } = modelsConfig[model];
        let noise = (Math.random() - 0.5) * vol * 2;
        
        let modelShock = marketShock;
        if (model === 'VLSTM' || model === 'LPatchTST') modelShock *= 0.4;
        if (model === 'xLSTM') modelShock *= 0.6;
        if (isCrisis) modelShock = -0.15 * (model === 'Passive (S&P500)' ? 1 : 0.4);

        const monthlyReturn = mean + noise + modelShock;
        currentValues[model] = currentValues[model] * (1 + monthlyReturn);
        
        dataPoint[model] = currentValues[model];
        dataPoint.rawReturns[model] = monthlyReturn;
      }
    });
    data.push(dataPoint);
  }
  return data;
};
