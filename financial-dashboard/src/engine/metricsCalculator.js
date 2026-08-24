import { calculateCAGR, calculateSharpe, calculateDrawdown } from '../utils/mathHelpers';

export const calculateAllMetrics = (data) => {
  if (!data || data.length === 0) return [];
  
  const models = Object.keys(data[0]).filter(k => k !== 'date' && k !== 'displayDate' && k !== 'year' && k !== 'rawReturns');
  const yearsTotal = data.length / 12;

  return models.map(model => {
    const values = data.map(d => d[model]);
    const returns = data.slice(1).map(d => d.rawReturns[model]);
    
    const finalValue = values[values.length - 1];
    const cagr = calculateCAGR(1.0, finalValue, yearsTotal);
    const sharpe = calculateSharpe(returns);
    
    const drawdowns = calculateDrawdown(values);
    const maxDrawdown = Math.min(...drawdowns);
    
    const positiveMonths = returns.filter(r => r > 0).length;
    const winRate = positiveMonths / returns.length;

    const variance = returns.reduce((a, b) => a + Math.pow(b - (returns.reduce((sum, r)=>sum+r,0)/returns.length), 2), 0) / returns.length;
    const annVolatility = Math.sqrt(variance) * Math.sqrt(12);

    return {
      model,
      finalMultiplier: finalValue,
      cagr: cagr * 100,
      sharpe: sharpe,
      maxDrawdown: maxDrawdown * 100,
      winRate: winRate * 100,
      volatility: annVolatility * 100,
      drawdowns
    };
  });
};
