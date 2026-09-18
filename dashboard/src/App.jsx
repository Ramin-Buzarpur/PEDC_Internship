import React, { useState, useMemo, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ReferenceLine } from 'recharts';
import { Activity, BarChart2, Info } from 'lucide-react';

import { generateMockData } from './data/mockData';
import { calculateAllMetrics } from './engine/metricsCalculator';
import { getRecommendations } from './engine/recommender';
import { RecommendationPanel } from './components/RecommendationPanel';

export default function App() {
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [recs, setRecs] = useState(null);
  const [selectedModels, setSelectedModels] = useState(['VLSTM', 'xLSTM', 'Passive (S&P500)']);
  const [timeView, setTimeView] = useState('monthly');

  const allModels = ['Passive (S&P500)', 'AR1x', 'LSTM', 'xLSTM', 'LPatchTST', 'VLSTM'];
  
  const colors = {
    'Passive (S&P500)': '#9ca3af',
    'AR1x': '#ef4444',
    'LSTM': '#f59e0b',
    'xLSTM': '#8b5cf6',
    'LPatchTST': '#ec4899',
    'VLSTM': '#3b82f6'
  };

  useEffect(() => {
    const generatedData = generateMockData();
    setData(generatedData);
    
    const calculatedMetrics = calculateAllMetrics(generatedData);
    setMetrics(calculatedMetrics);
    setRecs(getRecommendations(calculatedMetrics));
  }, []);

  const toggleModel = (model) => {
    setSelectedModels(prev => 
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };

  const chartData = useMemo(() => {
    if (timeView === 'monthly') return data;
    const yearlyMap = new Map();
    data.forEach(d => { yearlyMap.set(d.year, d); });
    return Array.from(yearlyMap.values());
  }, [data, timeView]);

  const drawdownData = useMemo(() => {
    if (data.length === 0) return [];
    let peaks = {};
    allModels.forEach(m => peaks[m] = -Infinity);

    return data.map(d => {
      const ddPoint = { displayDate: d.displayDate };
      allModels.forEach(m => {
        if (d[m] > peaks[m]) peaks[m] = d[m];
        ddPoint[m] = peaks[m] === 0 ? 0 : ((d[m] - peaks[m]) / peaks[m]) * 100;
      });
      return ddPoint;
    });
  }, [data]);

  if (data.length === 0) return <div className="min-h-screen bg-gray-950 flex items-center justify-center text-white">در حال بارگذاری...</div>;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-200 p-4 md:p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center border-b border-gray-800 pb-6" dir="rtl">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-2">
              <Activity className="text-blue-500" />
              داشبورد تحلیلی مدل‌های دیپ‌لرنینگ مالی
            </h1>
            <p className="text-gray-400">بررسی و مقایسه سود تجمیعی، Drawdown و شاخص‌های ریسک</p>
          </div>
          
          <div className="mt-4 md:mt-0 flex gap-2">
            <button onClick={() => setTimeView('monthly')} className={`px-4 py-2 rounded-lg text-sm transition-colors ${timeView === 'monthly' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}>ماهانه</button>
            <button onClick={() => setTimeView('yearly')} className={`px-4 py-2 rounded-lg text-sm transition-colors ${timeView === 'yearly' ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}>سالانه</button>
          </div>
        </header>

        <RecommendationPanel recs={recs} />

        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5" dir="rtl">
          <h3 className="text-gray-400 mb-3 text-sm font-semibold flex items-center gap-2">
            <BarChart2 size={16} /> انتخاب مدل‌ها برای نمایش در نمودارها
          </h3>
          <div className="flex flex-wrap gap-2">
            {allModels.map(model => (
              <button key={model} onClick={() => toggleModel(model)} className={`px-4 py-2 rounded-full text-sm font-medium border transition-all ${selectedModels.includes(model) ? 'bg-gray-800 text-white border-gray-600' : 'bg-transparent text-gray-500 border-gray-800'}`}>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full" style={{ backgroundColor: colors[model] }}></span>
                  {model}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 shadow-lg">
            <h3 className="text-white text-lg font-semibold mb-4 text-right">سود تجمیعی</h3>
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="displayDate" stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} />
                  <YAxis stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} domain={['dataMin - 0.2', 'dataMax + 0.5']} tickFormatter={(val) => val.toFixed(1) + 'x'}/>
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff', borderRadius: '8px' }} />
                  <Legend verticalAlign="top" height={36} />
                  {selectedModels.map(model => (
                    <Line key={model} type="monotone" dataKey={model} stroke={colors[model]} strokeWidth={2} dot={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 shadow-lg">
            <h3 className="text-white text-lg font-semibold mb-4 text-right">افت سرمایه (Drawdown)</h3>
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={drawdownData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="displayDate" stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} />
                  <YAxis stroke="#9ca3af" tick={{fill: '#9ca3af', fontSize: 12}} tickFormatter={(val) => val.toFixed(0) + '%'}/>
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#fff', borderRadius: '8px' }} />
                  <Legend verticalAlign="top" height={36} />
                  <ReferenceLine y={0} stroke="#4b5563" />
                  {selectedModels.map(model => (
                    <Area key={model} type="monotone" dataKey={model} stroke={colors[model]} fill={colors[model]} fillOpacity={0.1} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-lg" dir="rtl">
          <div className="p-5 border-b border-gray-800 flex justify-between items-center">
            <h3 className="text-white text-lg font-semibold flex items-center gap-2">
              <Info size={20} className="text-gray-400" /> جدول جامع شاخص‌ها
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-right text-sm">
              <thead className="bg-gray-800/50 text-gray-400">
                <tr>
                  <th className="p-4">مدل استراتژی</th>
                  <th className="p-4">نسبت شارپ (SR)</th>
                  <th className="p-4">سود سالانه (CAGR)</th>
                  <th className="p-4">بیشترین افت (Max DD)</th>
                  <th className="p-4">نوسانات (Volatility)</th>
                  <th className="p-4">درصد برد (Win Rate)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {metrics.sort((a, b) => b.sharpe - a.sharpe).map((m) => (
                  <tr key={m.model} className="hover:bg-gray-800/30 transition-colors">
                    <td className="p-4 font-medium flex items-center gap-2 text-white">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: colors[m.model] }}></span>
                      {m.model}
                    </td>
                    <td className="p-4 text-blue-400 font-bold">{m.sharpe.toFixed(2)}</td>
                    <td className="p-4 text-green-400">{m.cagr.toFixed(1)}%</td>
                    <td className="p-4 text-red-400">{m.maxDrawdown.toFixed(1)}%</td>
                    <td className="p-4 text-gray-300">{m.volatility.toFixed(1)}%</td>
                    <td className="p-4 text-purple-400">{m.winRate.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}
