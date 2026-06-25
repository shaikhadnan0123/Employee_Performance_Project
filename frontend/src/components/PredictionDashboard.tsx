import React, { useState } from 'react';
import { 
  Gauge, Users, Calendar, Briefcase, Clock, 
  Coins, Sparkles, Cpu, AlertCircle, 
  ArrowRight, Layers, Activity, RefreshCw 
} from 'lucide-react';

interface PredictionResult {
  success: boolean;
  prediction_value: number;
  prediction_text: string;
  prediction_level: 'average' | 'medium' | 'high';
  error_message?: string;
}

export const PredictionDashboard: React.FC = () => {
  const [formData, setFormData] = useState({
    quarter: 1,
    department: 'sweing',
    day: 'Monday',
    team: 1,
    targeted_productivity: 0.80,
    smv: 11.55,
    wip: 500.0,
    over_time: 0,
    incentive: 0,
    idle_time: 0.0,
    idle_men: 0,
    no_of_style_change: 0,
    no_of_workers: 15.0
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const isNumber = [
      'quarter', 'team', 'targeted_productivity', 'smv', 'wip', 
      'over_time', 'incentive', 'idle_time', 'idle_men', 
      'no_of_style_change', 'no_of_workers'
    ].includes(name);

    setFormData(prev => ({
      ...prev,
      [name]: isNumber ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    // Give a slight simulated delay for the geological drilling effect
    await new Promise(resolve => setTimeout(resolve, 800));

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setResult(data);
      } else {
        setError(data.error_message || 'Failed to compute productivity prediction.');
      }
    } catch (err: any) {
      setError('Could not connect to prediction service. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      quarter: 1,
      department: 'sweing',
      day: 'Monday',
      team: 1,
      targeted_productivity: 0.80,
      smv: 11.55,
      wip: 500.0,
      over_time: 0,
      incentive: 0,
      idle_time: 0.0,
      idle_men: 0,
      no_of_style_change: 0,
      no_of_workers: 15.0
    });
    setResult(null);
    setError(null);
  };

  // Get color styles based on prediction output level
  const getLevelStyles = (level: 'average' | 'medium' | 'high') => {
    switch (level) {
      case 'high':
        return {
          glow: 'shadow-[#10b981]/20 border-[#10b981]/30',
          text: 'text-[#10b981]',
          bg: 'bg-[#10b981]/10',
          circle: 'stroke-[#10b981]',
          status: 'Optimal Output'
        };
      case 'medium':
        return {
          glow: 'shadow-[#f59e0b]/20 border-[#f59e0b]/30',
          text: 'text-[#f59e0b]',
          bg: 'bg-[#f59e0b]/10',
          circle: 'stroke-[#f59e0b]',
          status: 'Standard Output'
        };
      default:
        return {
          glow: 'shadow-[#ef4444]/20 border-[#ef4444]/30',
          text: 'text-[#ef4444]',
          bg: 'bg-[#ef4444]/10',
          circle: 'stroke-[#ef4444]',
          status: 'Sub-optimal Output'
        };
    }
  };

  return (
    <div id="dig-dashboard" className="w-full min-h-screen bg-[#060606] text-white py-20 px-4 sm:px-6 lg:px-8 border-t border-white/10 relative overflow-hidden">
      {/* Decorative geological grid lines */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#111_1px,transparent_1px),linear-gradient(to_bottom,#111_1px,transparent_1px)] bg-[size:4rem_4rem] pointer-events-none opacity-20" />
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-[#e8702a]/5 rounded-full filter blur-[120px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-[#e8702a]/3 rounded-full filter blur-[100px] pointer-events-none" />

      <div className="max-w-6xl mx-auto relative z-10">
        
        {/* Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold text-[#e8702a] uppercase tracking-wider mb-4">
            <Layers className="w-3.5 h-3.5" /> Lithos deep time analytics
          </div>
          <h2 className="text-3xl sm:text-5xl font-playfair italic mb-4">
            Peel Back the Performance Layers
          </h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-sm sm:text-base leading-relaxed">
            Just as geological layers trace millions of years of sediment, our performance model cuts through workforce datasets to predict team productivity levels in real time.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Form Side */}
          <div className="lg:col-span-7 bg-[#0b0b0b] border border-white/10 rounded-3xl p-6 sm:p-8 shadow-2xl relative">
            <div className="absolute top-0 left-10 transform -translate-y-1/2 bg-[#e8702a] text-black text-xs font-bold px-3 py-1 rounded-md uppercase tracking-widest">
              Parameters
            </div>
            
            <form onSubmit={handleSubmit} className="space-y-6">
              
              {/* Section 1: Temporal & Organizational Context */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#e8702a] mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#e8702a]" /> Organizational Layer
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-gray-500" /> Quarter
                    </label>
                    <select
                      name="quarter"
                      value={formData.quarter}
                      onChange={handleChange}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                    >
                      <option className="bg-[#0b0b0b]" value={1}>Quarter 1</option>
                      <option className="bg-[#0b0b0b]" value={2}>Quarter 2</option>
                      <option className="bg-[#0b0b0b]" value={3}>Quarter 3</option>
                      <option className="bg-[#0b0b0b]" value={4}>Quarter 4</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Briefcase className="w-3.5 h-3.5 text-gray-500" /> Department
                    </label>
                    <select
                      name="department"
                      value={formData.department}
                      onChange={handleChange}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                    >
                      <option className="bg-[#0b0b0b]" value="sweing">Sewing Department</option>
                      <option className="bg-[#0b0b0b]" value="finishing">Finishing Department</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Calendar className="w-3.5 h-3.5 text-gray-500" /> Day of Week
                    </label>
                    <select
                      name="day"
                      value={formData.day}
                      onChange={handleChange}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                    >
                      {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].map(d => (
                        <option key={d} className="bg-[#0b0b0b]" value={d}>{d}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-gray-500" /> Team Number
                    </label>
                    <input
                      type="number"
                      name="team"
                      value={formData.team}
                      onChange={handleChange}
                      min={1}
                      max={20}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Section 2: Complexity & Workforce Density */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#e8702a] mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#e8702a]" /> Target & Workforce Specs
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center justify-between">
                      <span className="flex items-center gap-1.5">
                        <Gauge className="w-3.5 h-3.5 text-gray-500" /> Target Productivity
                      </span>
                      <span className="text-[#e8702a] font-mono">{formData.targeted_productivity.toFixed(2)}</span>
                    </label>
                    <input
                      type="range"
                      name="targeted_productivity"
                      value={formData.targeted_productivity}
                      onChange={handleChange}
                      min={0.1}
                      max={1.0}
                      step={0.01}
                      className="w-full accent-[#e8702a] h-1 bg-white/10 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center justify-between">
                      <span className="flex items-center gap-1.5">
                        <Cpu className="w-3.5 h-3.5 text-gray-500" /> SMV (Standard Minute Value)
                      </span>
                      <span className="text-white font-mono">{formData.smv}</span>
                    </label>
                    <input
                      type="number"
                      name="smv"
                      value={formData.smv}
                      onChange={handleChange}
                      step={0.01}
                      min={1}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Activity className="w-3.5 h-3.5 text-gray-500" /> WIP (Work in Progress)
                    </label>
                    <input
                      type="number"
                      name="wip"
                      value={formData.wip}
                      onChange={handleChange}
                      step={1}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-gray-500" /> Team Size (No. of Workers)
                    </label>
                    <input
                      type="number"
                      name="no_of_workers"
                      value={formData.no_of_workers}
                      onChange={handleChange}
                      step={0.5}
                      min={1}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Section 3: Time & Incentives */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#e8702a] mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#e8702a]" /> Motivators & Duration
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-gray-500" /> Overtime (Minutes)
                    </label>
                    <input
                      type="number"
                      name="over_time"
                      value={formData.over_time}
                      onChange={handleChange}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Coins className="w-3.5 h-3.5 text-gray-500" /> Financial Incentive
                    </label>
                    <input
                      type="number"
                      name="incentive"
                      value={formData.incentive}
                      onChange={handleChange}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-4 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Section 4: Interruptions */}
              <div>
                <h3 className="text-xs font-bold uppercase tracking-widest text-[#e8702a] mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#e8702a]" /> Friction Layers
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Clock className="w-3.5 h-3.5 text-gray-500" /> Idle Time
                    </label>
                    <input
                      type="number"
                      name="idle_time"
                      value={formData.idle_time}
                      onChange={handleChange}
                      step={0.1}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-3 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-gray-500" /> Idle Men
                    </label>
                    <input
                      type="number"
                      name="idle_men"
                      value={formData.idle_men}
                      onChange={handleChange}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-3 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-400 mb-1.5 flex items-center gap-1.5">
                      <RefreshCw className="w-3.5 h-3.5 text-gray-500" /> Style Changes
                    </label>
                    <input
                      type="number"
                      name="no_of_style_change"
                      value={formData.no_of_style_change}
                      onChange={handleChange}
                      min={0}
                      className="w-full bg-white/5 border border-white/10 focus:border-[#e8702a] rounded-xl px-3 py-2 text-sm outline-none transition-all"
                      required
                    />
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 pt-4">
                <button
                  type="button"
                  onClick={resetForm}
                  className="w-1/3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full py-3.5 text-sm font-semibold transition-all active:scale-[0.98]"
                >
                  Reset Defaults
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="w-2/3 bg-[#e8702a] hover:bg-[#d2611f] disabled:bg-[#e8702a]/50 text-white rounded-full py-3.5 text-sm font-bold shadow-lg hover:shadow-[#e8702a]/30 transition-all hover:scale-[1.01] active:scale-[0.98] flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Digging through data...
                    </>
                  ) : (
                    <>
                      Compute Productivity
                      <ArrowRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>

          {/* Outcome Side */}
          <div className="lg:col-span-5 h-full space-y-6">
            
            {/* Dynamic Output Card */}
            <div className="bg-[#0b0b0b] border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden min-h-[460px] flex flex-col justify-between">
              <div className="absolute top-0 right-0 w-[200px] h-[200px] bg-white/2 rounded-full filter blur-[50px] pointer-events-none" />
              
              <div className="absolute top-0 left-10 transform -translate-y-1/2 bg-white/10 text-white text-xs font-semibold px-3 py-1 rounded-md uppercase tracking-wider">
                ML Assessment
              </div>

              {loading ? (
                /* Drilling/Loader State */
                <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 rounded-full border-4 border-dashed border-[#e8702a] animate-spin mb-6" />
                  <h4 className="text-lg font-semibold text-white mb-2">Analyzing Performance Crust</h4>
                  <p className="text-xs text-gray-400 max-w-xs">
                    Transforming categorical elements, parsing SMV factors, and executing model calculations...
                  </p>
                </div>
              ) : result ? (
                /* Success State */
                (() => {
                  const styles = getLevelStyles(result.prediction_level);
                  const circleCircumference = 2 * Math.PI * 45;
                  const strokeDashoffset = circleCircumference - (result.prediction_value * circleCircumference);

                  return (
                    <div className="flex-1 flex flex-col justify-between h-full space-y-8 animate-fade-in">
                      
                      {/* Gauge Indicator */}
                      <div className="flex flex-col items-center py-4">
                        <div className="relative w-36 h-36 flex items-center justify-center">
                          {/* Background circle */}
                          <svg className="w-full h-full transform -rotate-90">
                            <circle
                              cx="72"
                              cy="72"
                              r="45"
                              className="stroke-white/5"
                              strokeWidth="8"
                              fill="transparent"
                            />
                            {/* Animated indicator path */}
                            <circle
                              cx="72"
                              cy="72"
                              r="45"
                              className={`${styles.circle} transition-all duration-1000 ease-out`}
                              strokeWidth="8"
                              strokeDasharray={circleCircumference}
                              strokeDashoffset={strokeDashoffset}
                              strokeLinecap="round"
                              fill="transparent"
                            />
                          </svg>
                          <div className="absolute flex flex-col items-center">
                            <span className="text-3xl font-bold tracking-tight">{result.prediction_value.toFixed(4)}</span>
                            <span className="text-[10px] text-gray-500 uppercase tracking-widest">Productivity</span>
                          </div>
                        </div>
                        
                        <div className={`mt-4 px-3 py-1 rounded-full text-xs font-semibold ${styles.bg} ${styles.text} border ${styles.glow}`}>
                          {styles.status}
                        </div>
                      </div>

                      {/* Summary text */}
                      <div className="bg-white/5 border border-white/5 rounded-2xl p-5 text-center">
                        <p className="text-gray-300 text-sm italic">
                          "{result.prediction_text}"
                        </p>
                      </div>

                      {/* Explanation */}
                      <div className="text-xs text-gray-400 space-y-2">
                        <div className="flex justify-between border-b border-white/5 pb-2">
                          <span>Model Algorithm:</span>
                          <span className="text-white font-semibold">XGBoost Regressor</span>
                        </div>
                        <div className="flex justify-between border-b border-white/5 pb-2">
                          <span>Target Objective:</span>
                          <span className="text-white font-semibold">{formData.targeted_productivity.toFixed(2)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Achieved Ratio:</span>
                          <span className={`${result.prediction_value >= formData.targeted_productivity ? 'text-[#10b981]' : 'text-[#ef4444]'} font-semibold`}>
                            {((result.prediction_value / formData.targeted_productivity) * 100).toFixed(1)}% of Target
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })()
              ) : error ? (
                /* Error State */
                <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-12 h-12 bg-red-500/10 border border-red-500/20 rounded-full flex items-center justify-center mb-6 text-red-500">
                    <AlertCircle className="w-6 h-6" />
                  </div>
                  <h4 className="text-lg font-semibold text-white mb-2">Analysis Failed</h4>
                  <p className="text-xs text-gray-400 max-w-xs mb-6">
                    {error}
                  </p>
                  <button
                    onClick={handleSubmit}
                    className="bg-white/5 hover:bg-white/10 text-white border border-white/10 text-xs px-4 py-2 rounded-full transition-all"
                  >
                    Retry Analysis
                  </button>
                </div>
              ) : (
                /* Default State */
                <div className="flex-1 flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-12 h-12 bg-[#e8702a]/10 border border-[#e8702a]/20 rounded-full flex items-center justify-center mb-6 text-[#e8702a]">
                    <Sparkles className="w-6 h-6" />
                  </div>
                  <h4 className="text-lg font-semibold text-white mb-2">Ready for Geological Digging</h4>
                  <p className="text-xs text-gray-400 max-w-xs leading-relaxed">
                    Adjust the parameters on the left (such as Team number, Target, and Overtime) and press "Compute Productivity" to run the ML algorithm.
                  </p>
                </div>
              )}
            </div>

            {/* Quick Context Card */}
            <div className="bg-[#0b0b0b] border border-white/10 rounded-3xl p-6 text-xs text-gray-400 leading-relaxed relative overflow-hidden">
              <h5 className="font-bold text-white uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-[#e8702a]" />
                Sedimentary Metaphor & ML
              </h5>
              In geology, the thickness and composition of layers are shaped by variables like temperature, pressure, and biological material. Similarly, employee output is layered by variables like targeted productivity, team sizes, and overtime. Our ML model is trained to parse this performance crust and forecast structural productivity.
            </div>

          </div>

        </div>

      </div>
    </div>
  );
};
