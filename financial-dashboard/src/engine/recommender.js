import React from 'react';
import { CheckCircle } from 'lucide-react';

export const RecommendationPanel = ({ recs }) => {
  if (!recs) return null;
  return (
    <div className="bg-gradient-to-r from-blue-900/40 to-indigo-900/40 border border-blue-800/50 rounded-xl p-6 mb-6" dir="rtl">
      <div className="flex items-center gap-3 mb-4">
        <CheckCircle className="text-blue-400 w-6 h-6" />
        <h2 className="text-xl font-bold text-white">پیشنهاد سیستم هوشمند</h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
          <div className="text-sm text-gray-400 mb-2">🏆 بهترین عملکرد کلی (نسبت شارپ)</div>
          <div className="text-lg font-bold text-blue-400">{recs.overallBest.model}</div>
          <p className="text-xs text-gray-500 mt-2">بالاترین تعادل بین سود و ریسک با شارپ {recs.overallBest.sharpe.toFixed(2)}</p>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
          <div className="text-sm text-gray-400 mb-2">🛡️ کم‌ریسک‌ترین (کمترین افت سرمایه)</div>
          <div className="text-lg font-bold text-green-400">{recs.safest.model}</div>
          <p className="text-xs text-gray-500 mt-2">مقاوم‌ترین در برابر ریزش‌های بازار ({recs.safest.maxDrawdown.toFixed(1)}%)</p>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700/50">
          <div className="text-sm text-gray-400 mb-2">🚀 بیشترین سود (CAGR)</div>
          <div className="text-lg font-bold text-purple-400">{recs.highestReturn.model}</div>
          <p className="text-xs text-gray-500 mt-2">بالاترین سود سالانه ترکیبی با {recs.highestReturn.cagr.toFixed(1)}%</p>
        </div>
      </div>
    </div>
  );
};
