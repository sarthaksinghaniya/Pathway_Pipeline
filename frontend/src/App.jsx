import React, { useState, useEffect } from 'react';
import { BarChart3, FileText, Info } from 'lucide-react';
import Dashboard from './components/Dashboard';
import ResultsTable from './components/ResultsTable';
import DetailView from './components/DetailView';
import Methodology from './components/Methodology';
import { loadResults, loadEvidence } from './utils/dataLoader';

const App = () => {
  const [currentView, setCurrentView] = useState('dashboard');
  const [data, setData] = useState([]);
  const [summary, setSummary] = useState({});
  const [evidence, setEvidence] = useState({});
  const [selectedItem, setSelectedItem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Load main results
        const resultsData = await loadResults();
        setData(resultsData.results);
        setSummary(resultsData.summary);
        
        // Load evidence if available
        const evidenceData = await loadEvidence();
        setEvidence(evidenceData);
        
      } catch (err) {
        console.error('Failed to load data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  const handleRowClick = (item) => {
    setSelectedItem(item);
    setCurrentView('detail');
  };

  const handleBackToResults = () => {
    setSelectedItem(null);
    setCurrentView('results');
  };

  const renderNavigation = () => (
    <nav className="bg-white shadow-sm border border-gray-200">
      <div className="page-container">
        <div className="flex items-center justify-between py-4">
          <div className="flex items-center space-x-8">
            <h1 className="text-xl font-bold text-gray-900 flex items-center">
              <FileText className="h-6 w-6 mr-2 text-blue-600" />
              Consistency Evaluation
            </h1>
          </div>
          
          <div className="flex items-center space-x-1">
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`px-4 py-2 text-sm font-medium rounded-l-lg transition-colors ${
                currentView === 'dashboard'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              Dashboard
            </button>
            <button
              onClick={() => setCurrentView('results')}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                currentView === 'results'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              Results
            </button>
            <button
              onClick={() => setCurrentView('methodology')}
              className={`px-4 py-2 text-sm font-medium rounded-r-lg transition-colors ${
                currentView === 'methodology'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }`}
            >
              Methodology
            </button>
          </div>
        </div>
      </div>
    </nav>
  );

  const renderContent = () => {
    if (loading) {
      return (
        <div className="page-container">
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <BarChart3 className="h-12 w-12 text-blue-600 mx-auto mb-4 animate-pulse" />
              <p className="text-lg font-medium text-gray-900">Loading results...</p>
              <p className="text-sm text-gray-600 mt-2">
                Please wait while we load the evaluation data.
              </p>
            </div>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="page-container">
          <div className="flex items-center justify-center py-20">
            <div className="text-center max-w-md">
              <Info className="h-12 w-12 text-red-600 mx-auto mb-4" />
              <h2 className="text-xl font-bold text-red-900 mb-2">Error Loading Data</h2>
              <p className="text-gray-600 mb-4">{error}</p>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      );
    }

    switch (currentView) {
      case 'dashboard':
        return <Dashboard data={data} summary={summary} />;
      case 'results':
        return <ResultsTable data={data} onRowClick={handleRowClick} />;
      case 'detail':
        return (
          <DetailView 
            item={selectedItem} 
            evidence={evidence} 
            onBack={handleBackToResults} 
          />
        );
      case 'methodology':
        return <Methodology />;
      default:
        return <Dashboard data={data} summary={summary} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {renderNavigation()}
      {renderContent()}
      
      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="page-container py-6">
          <div className="text-center text-sm text-gray-600">
            <p>
              Read-only visualization interface for Backstory–Novel Consistency Evaluation
            </p>
            <p className="mt-1">
              This interface does not modify or recompute any ML predictions
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
