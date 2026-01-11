import React from 'react';
import { BookOpen, Search, Scale, Shield, Code, Database } from 'lucide-react';

const Methodology = () => {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Transparency & Methodology
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Understanding how the consistency evaluation system works and the design principles behind it.
        </p>
      </div>

      {/* System Overview */}
      <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-200">
        <h2 className="text-2xl font-semibold text-gray-900 mb-6">System Architecture</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="p-4 bg-blue-100 rounded-lg inline-block mb-4">
              <BookOpen className="h-8 w-8 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Novel Corpus</h3>
            <p className="text-sm text-gray-600">
              Full-length novels are chunked into overlapping segments and indexed using semantic embeddings 
              with TF-IDF fallback for robustness.
            </p>
          </div>
          
          <div className="text-center">
            <div className="p-4 bg-green-100 rounded-lg inline-block mb-4">
              <Search className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Retrieval</h3>
            <p className="text-sm text-gray-600">
              For each backstory, the system retrieves the most semantically relevant novel chunks 
              using similarity search over the indexed corpus.
            </p>
          </div>
          
          <div className="text-center">
            <div className="p-4 bg-purple-100 rounded-lg inline-block mb-4">
              <Scale className="h-8 w-8 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Classification</h3>
            <p className="text-sm text-gray-600">
              Retrieved evidence is analyzed for supporting or contradictory signals, then aggregated 
              using deterministic rules to produce a binary verdict.
            </p>
          </div>
        </div>
      </div>

      {/* Technical Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Retrieval Process */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Search className="h-6 w-6 mr-2 text-blue-600" />
            Retrieval-Augmented Reasoning
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">1. Text Chunking</h4>
              <p className="text-sm text-gray-600">
                Novels are segmented into 1000-token chunks with 150-token overlap, ensuring 
                context preservation while maintaining manageable processing units.
              </p>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-2">2. Semantic Indexing</h4>
              <p className="text-sm text-gray-600">
                Sentence transformers create dense vector representations, with TF-IDF fallback 
                for environments without GPU support.
              </p>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-2">3. Similarity Search</h4>
              <p className="text-sm text-gray-600">
                Cosine similarity identifies the most relevant novel passages for each backstory, 
                typically retrieving 5-10 chunks for analysis.
              </p>
            </div>
          </div>
        </div>

        {/* Classification Logic */}
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Shield className="h-6 w-6 mr-2 text-green-600" />
            Evidence-Based Decision Making
          </h3>
          
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Evidence Classification</h4>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-sm"><strong>SUPPORT:</strong> Evidence aligns with backstory</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span className="text-sm"><strong>CONTRADICT:</strong> Evidence conflicts with backstory</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                  <span className="text-sm"><strong>NEUTRAL:</strong> Neither strong support nor contradiction</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium text-gray-900 mb-2">Deterministic Aggregation</h4>
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div className="space-y-2 text-sm">
                  <div className="flex items-center space-x-2">
                    <Code className="h-4 w-4 text-gray-600" />
                    <span><strong>Rule:</strong> Any strong contradiction (confidence ≥ 0.7) → Inconsistent</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Code className="h-4 w-4 text-gray-600" />
                    <span><strong>Default:</strong> No strong contradictions → Consistent</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Design Principles */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Design Principles</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Safety & Reliability</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start space-x-2">
                <Shield className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>Read-only interface - no model modification</span>
              </li>
              <li className="flex items-start space-x-2">
                <Database className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Deterministic outputs - reproducible results</span>
              </li>
              <li className="flex items-start space-x-2">
                <Code className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                <span>Evidence-based decisions - explainable reasoning</span>
              </li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Technical Excellence</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start space-x-2">
                <BookOpen className="h-4 w-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Robust fallback mechanisms (TF-IDF)</span>
              </li>
              <li className="flex items-start space-x-2">
                <Search className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>Semantic understanding with context</span>
              </li>
              <li className="flex items-start space-x-2">
                <Scale className="h-4 w-4 text-purple-600 mt-0.5 flex-shrink-0" />
                <span>Scalable architecture for large corpora</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Important Notice */}
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-red-900 mb-3">Important Notice</h3>
        <div className="space-y-2 text-sm text-red-800">
          <p>
            <strong>This interface is strictly for visualization and inspection purposes.</strong>
          </p>
          <ul className="ml-4 list-disc space-y-1">
            <li>It does not modify, retrain, or recompute any ML models</li>
            <li>It displays pre-generated results from frozen backend computations</li>
            <li>No user input affects the underlying predictions</li>
            <li>All decisions are based on evidence from the original novels</li>
          </ul>
        </div>
      </div>

      {/* Technical Stack */}
      <div className="bg-gray-50 p-6 rounded-lg border border-gray-200">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Implementation Stack</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-2">Frontend</h4>
            <p className="text-sm text-gray-600">React + Vite</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-2">Styling</h4>
            <p className="text-sm text-gray-600">Tailwind CSS</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-2">Charts</h4>
            <p className="text-sm text-gray-600">Recharts</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <h4 className="font-medium text-gray-900 mb-2">Data</h4>
            <p className="text-sm text-gray-600">CSV/JSON (Read-only)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Methodology;
