import React from 'react';
import { ArrowLeft, CheckCircle, XCircle, MinusCircle, Search } from 'lucide-react';
import { getPredictionLabel, getPredictionClass } from '../utils/dataLoader';

const DetailView = ({ item, evidence, onBack }) => {
  const getEvidenceIcon = (label) => {
    switch (label?.toUpperCase()) {
      case 'SUPPORT':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'CONTRADICT':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'NEUTRAL':
        return <MinusCircle className="h-5 w-5 text-gray-600" />;
      default:
        return <MinusCircle className="h-5 w-5 text-gray-600" />;
    }
  };

  const getEvidenceClass = (label) => {
    switch (label?.toUpperCase()) {
      case 'SUPPORT':
        return 'support';
      case 'CONTRADICT':
        return 'contradict';
      case 'NEUTRAL':
        return 'neutral';
      default:
        return 'neutral';
    }
  };

  const evidenceData = evidence[item.id] || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
        >
          <ArrowLeft className="h-5 w-5" />
          <span>Back to Results</span>
        </button>
        
        <div className={`prediction-badge ${getPredictionClass(item.prediction)}`}>
          {getPredictionLabel(item.prediction)}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Backstory Section */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Backstory Analysis</h2>
            
            <div className="space-y-4">
              {/* Metadata */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm text-gray-600">ID</p>
                  <p className="font-medium text-gray-900">{item.id}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Prediction</p>
                  <p className={`font-medium ${getPredictionClass(item.prediction) === 'consistent' ? 'text-green-600' : 'text-red-600'}`}>
                    {getPredictionLabel(item.prediction)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Length</p>
                  <p className="font-medium text-gray-900">
                    {item.backstory_length || (item.backstory ? item.backstory.length : 'N/A')}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Chunks Retrieved</p>
                  <p className="font-medium text-gray-900">{item.chunks_retrieved || 'N/A'}</p>
                </div>
              </div>

              {/* Full Backstory */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Complete Backstory</h3>
                <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-gray-700 whitespace-pre-wrap leading-relaxed">
                    {item.backstory || 'No backstory available'}
                  </p>
                </div>
              </div>

              {/* Processing Notes */}
              {item.error && (
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h4 className="font-medium text-yellow-800 mb-2">Processing Note</h4>
                  <p className="text-sm text-yellow-700">{item.error}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Evidence Section */}
        <div className="space-y-4">
          <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Retrieved Evidence</h2>
            
            {evidenceData.length > 0 ? (
              <div className="space-y-3">
                {evidenceData.map((evidence, index) => (
                  <div key={index} className={`evidence-card ${getEvidenceClass(evidence.label)}`}>
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-1">
                        {getEvidenceIcon(evidence.label)}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-gray-900">
                            {evidence.label || 'NEUTRAL'}
                          </h4>
                          {evidence.similarity && (
                            <span className="text-sm text-gray-600 bg-white px-2 py-1 rounded border">
                              {evidence.similarity.toFixed(3)}
                            </span>
                          )}
                        </div>
                        
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {evidence.text}
                        </p>
                        
                        {evidence.source && (
                          <p className="text-xs text-gray-500 italic">
                            Source: {evidence.source}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No evidence data available</p>
                <p className="text-sm text-gray-500 mt-2">
                  Evidence may not have been generated for this prediction.
                </p>
              </div>
            )}
          </div>

          {/* Decision Explanation */}
          <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
            <h3 className="text-lg font-semibold text-blue-900 mb-3">Decision Logic</h3>
            <div className="space-y-3 text-sm text-blue-800">
              <p>
                <strong>Retrieval:</strong> System searched the full novel text for semantically 
                similar passages to the backstory.
              </p>
              <p>
                <strong>Analysis:</strong> Each retrieved chunk was evaluated for supporting or 
                contradictory evidence.
              </p>
              <p>
                <strong>Aggregation:</strong> Deterministic rules were applied:
              </p>
              <ul className="ml-4 list-disc space-y-1">
                <li>Any strong contradiction → <span className="font-medium">Inconsistent</span></li>
                <li>No strong contradictions → <span className="font-medium">Consistent</span></li>
              </ul>
              <div className="mt-4 p-3 bg-white rounded border border-blue-200">
                <p className="font-medium">
                  Final Verdict: <span className={getPredictionClass(item.prediction) === 'consistent' ? 'text-green-700' : 'text-red-700'}>
                    {getPredictionLabel(item.prediction)}
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetailView;
