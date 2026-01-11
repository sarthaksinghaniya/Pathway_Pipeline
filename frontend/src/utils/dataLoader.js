/**
 * Data Loading Utilities
 * 
 * This module handles loading and processing of prediction results.
 * It is READ-ONLY and does not modify any data.
 */

/**
 * Load results from CSV or JSON file
 * @param {string} filePath - Path to the results file
 * @returns {Promise<Object>} - Parsed results data
 */
export async function loadResults(filePath = '/results.csv') {
  try {
    // Try to determine file type from extension
    const isJson = filePath.endsWith('.json');
    
    if (isJson) {
      const response = await fetch(filePath);
      if (!response.ok) {
        throw new Error(`Failed to load JSON: ${response.statusText}`);
      }
      const data = await response.json();
      return processJsonData(data);
    } else {
      // Load CSV
      const response = await fetch(filePath);
      if (!response.ok) {
        throw new Error(`Failed to load CSV: ${response.statusText}`);
      }
      const csvText = await response.text();
      return parseCSV(csvText);
    }
  } catch (error) {
    console.error('Error loading results:', error);
    throw error;
  }
}

/**
 * Parse CSV text into structured data
 * @param {string} csvText - Raw CSV text
 * @returns {Object} - Parsed data
 */
function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const data = lines.slice(1).map(line => {
    const values = parseCSVLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || '';
    });
    return row;
  });

  return processCSVData(data);
}

/**
 * Parse a single CSV line handling quoted fields
 * @param {string} line - CSV line
 * @returns {Array} - Parsed values
 */
function parseCSVLine(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current.trim());
      current = '';
    } else {
      current += char;
    }
  }
  
  result.push(current.trim());
  return result;
}

/**
 * Process JSON data
 * @param {Object} data - Raw JSON data
 * @returns {Object} - Processed data
 */
function processJsonData(data) {
  if (Array.isArray(data)) {
    return {
      results: data,
      summary: calculateSummary(data),
      metadata: {
        source: 'json',
        loadedAt: new Date().toISOString(),
        totalRecords: data.length
      }
    };
  }
  
  return {
    results: data.results || [],
    summary: data.summary || {},
    metadata: {
      source: 'json',
      loadedAt: new Date().toISOString(),
      ...data.metadata
    }
  };
}

/**
 * Process CSV data
 * @param {Array} data - Parsed CSV data
 * @returns {Object} - Processed data
 */
function processCSVData(data) {
  return {
    results: data,
    summary: calculateSummary(data),
    metadata: {
      source: 'csv',
      loadedAt: new Date().toISOString(),
      totalRecords: data.length
    }
  };
}

/**
 * Calculate summary statistics
 * @param {Array} results - Results array
 * @returns {Object} - Summary statistics
 */
function calculateSummary(results) {
  const total = results.length;
  const consistent = results.filter(r => r.prediction === '1' || r.prediction === 1).length;
  const inconsistent = total - consistent;
  
  const avgLength = results.reduce((sum, r) => {
    const length = parseInt(r.backstory_length) || r.backstory?.length || 0;
    return sum + length;
  }, 0) / total;
  
  return {
    totalSamples: total,
    consistentCount: consistent,
    inconsistentCount: inconsistent,
    consistentPercentage: total > 0 ? (consistent / total) * 100 : 0,
    inconsistentPercentage: total > 0 ? (inconsistent / total) * 100 : 0,
    averageBackstoryLength: Math.round(avgLength),
    chunksRetrievedAvg: results.reduce((sum, r) => sum + (parseInt(r.chunks_retrieved) || 0), 0) / total
  };
}

/**
 * Load evidence data if available
 * @param {string} filePath - Path to evidence file
 * @returns {Promise<Object>} - Evidence data or empty object
 */
export async function loadEvidence(filePath = '/evidence.json') {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      console.warn('Evidence file not found, proceeding without evidence');
      return {};
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.warn('Error loading evidence:', error);
    return {};
  }
}

/**
 * Get prediction label text
 * @param {number|string} prediction - Prediction value (0 or 1)
 * @returns {string} - Human-readable label
 */
export function getPredictionLabel(prediction) {
  const pred = String(prediction);
  return pred === '1' ? 'Consistent' : 'Inconsistent';
}

/**
 * Get prediction color class
 * @param {number|string} prediction - Prediction value (0 or 1)
 * @returns {string} - CSS class
 */
export function getPredictionClass(prediction) {
  const pred = String(prediction);
  return pred === '1' ? 'consistent' : 'inconsistent';
}

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string} - Formatted number
 */
export function formatNumber(num) {
  return num.toLocaleString();
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} - Truncated text
 */
export function truncateText(text, maxLength = 100) {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}
