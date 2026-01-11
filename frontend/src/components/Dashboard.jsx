import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { FileText, TrendingUp, Users, Activity, AlertTriangle } from 'lucide-react';

const Dashboard = ({ data, summary }) => {
  const [accuracyData, setAccuracyData] = useState(null);

  useEffect(() => {
    // Load accuracy summary
    fetch('/accuracy_summary.json')
      .then(response => response.json())
      .then(data => setAccuracyData(data))
      .catch(error => console.log('Accuracy data not available:', error));
  }, []);

  const pieData = [
    { name: 'Consistent', value: summary.consistentCount, color: '#10b981' },
    { name: 'Inconsistent', value: summary.inconsistentCount, color: '#ef4444' }
  ];

  const barData = [
    { metric: 'Total Samples', value: summary.totalSamples },
    { metric: 'Avg Length', value: summary.averageBackstoryLength },
    { metric: 'Avg Chunks', value: Math.round(summary.chunksRetrievedAvg) }
  ];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border border-gray-200 rounded shadow-lg">
          <p className="text-sm font-medium">{`${label}: ${payload[0].value}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">
          Backstory–Novel Consistency Evaluation
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          A retrieval-augmented reasoning system that evaluates character backstories against full-length novels 
          for logical consistency and narrative coherence.
        </p>
      </div>

      {/* Model Performance Summary */}
      {accuracyData && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <FileText className="h-6 w-6 mr-2 text-blue-600" />
            Model Performance Summary
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="font-medium text-gray-900">Training Dataset</h4>
              <div className="space-y-2 text-sm">
                <p><strong>Accuracy:</strong> {accuracyData.train_results.accuracy_percentage.toFixed(1)}%</p>
                <p><strong>Examples:</strong> {accuracyData.train_results.total_examples}</p>
                <p><strong>Correct:</strong> {accuracyData.train_results.correct_predictions}</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="font-medium text-gray-900">Test Dataset</h4>
              <div className="space-y-2 text-sm">
                <p><strong>Consistent:</strong> {accuracyData.test_results.consistent_percentage.toFixed(1)}%</p>
                <p><strong>Inconsistent:</strong> {accuracyData.test_results.inconsistent_percentage.toFixed(1)}%</p>
                <p><strong>Examples:</strong> {accuracyData.test_results.total_examples}</p>
              </div>
            </div>
          </div>
          
          {accuracyData.overall_summary.model_behavior && (
            <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                <strong>Model Behavior:</strong> {accuracyData.overall_summary.model_behavior}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 rounded-lg">
              <FileText className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Total Samples</p>
              <p className="text-2xl font-semibold text-gray-900">
                {summary.totalSamples.toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 bg-green-100 rounded-lg">
              <TrendingUp className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Consistent</p>
              <p className="text-2xl font-semibold text-green-600">
                {summary.consistentPercentage.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 bg-red-100 rounded-lg">
              <Activity className="h-6 w-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Inconsistent</p>
              <p className="text-2xl font-semibold text-red-600">
                {summary.inconsistentPercentage.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        <div className="stat-card">
          <div className="flex items-center">
            <div className="p-3 bg-purple-100 rounded-lg">
              <Users className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Avg Length</p>
              <p className="text-2xl font-semibold text-gray-900">
                {summary.averageBackstoryLength}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Pie Chart */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System Explanation */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">How It Works</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">1. Retrieval</h4>
            <p className="text-sm text-gray-600">
              For each backstory, the system retrieves the most relevant novel chunks using semantic search 
              with sentence transformers and TF-IDF fallback.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">2. Analysis</h4>
            <p className="text-sm text-gray-600">
              Retrieved evidence is classified as SUPPORT, CONTRADICT, or NEUTRAL based on 
              linguistic signals and semantic similarity.
            </p>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">3. Decision</h4>
            <p className="text-sm text-gray-600">
              Deterministic aggregation rules convert the analysis into a binary verdict: 
              any strong contradiction → inconsistent, otherwise → consistent.
            </p>
          </div>
        </div>
        <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-800">
            <strong>Note:</strong> This interface displays pre-computed results and does not affect 
            the underlying predictions. All decisions are based on evidence from the original novels.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
