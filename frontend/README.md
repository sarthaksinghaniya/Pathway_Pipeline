# Frontend: Backstory–Novel Consistency Evaluation

A clean, professional React frontend for displaying ML evaluation results in a hackathon setting.

## 🎯 Purpose

This is a **READ-ONLY visualization layer** designed specifically for hackathon judges and evaluation. It displays pre-computed ML results without modifying any backend logic.

## ✨ Features

### 📊 Dashboard
- Summary statistics with key metrics
- Interactive charts (pie and bar charts)
- System explanation for judges
- Professional academic design

### 📋 Results Table
- Paginated, sortable table with 60+ predictions
- Search functionality
- Expandable rows for full backstory preview
- Color-coded prediction badges

### 🔍 Detail View
- Complete backstory analysis
- Retrieved evidence chunks with labels
- Decision logic explanation
- Evidence highlighting (Support/Contradict/Neutral)

### 📖 Methodology Section
- System architecture overview
- Technical implementation details
- Design principles and safety features
- Transparency documentation

## 🛠️ Tech Stack

- **Frontend**: React 18 + Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Data**: CSV/JSON (read-only)

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx      # Main dashboard with charts
│   │   ├── ResultsTable.jsx   # Sortable results table
│   │   ├── DetailView.jsx     # Individual result analysis
│   │   └── Methodology.jsx   # System explanation
│   ├── utils/
│   │   └── dataLoader.js     # CSV/JSON loading utilities
│   ├── App.jsx               # Main application
│   ├── main.jsx              # React entry point
│   └── index.css             # Tailwind styles
├── package.json
├── vite.config.js
├── tailwind.config.js
└── index.html
```

## 📊 Data Format

### Expected CSV Structure
```csv
id,prediction,backstory,backstory_length,chunks_retrieved,error
95,0,"Learning that Villefort meant to denounce him...",233,5,
136,0,"From 1800 onward he lived quietly...",141,5,
```

### Optional Evidence JSON
```json
{
  "95": [
    {
      "text": "Retrieved novel chunk...",
      "label": "CONTRADICT",
      "similarity": 0.847
    }
  ]
}
```

## 🎨 Design Principles

### ✅ Judge-Friendly
- Clear, non-technical explanations
- Professional academic appearance
- Intuitive navigation
- Mobile-responsive design

### 🔒 Safety Features
- **Read-only interface** - no model modification
- **No retraining** - displays frozen results
- **No database writes** - pure visualization
- **Transparent methodology** - explains all decisions

### 🎯 User Experience
- Smooth transitions and interactions
- Loading states and error handling
- Accessible design patterns
- Fast, responsive performance

## 📱 Pages & Navigation

1. **Dashboard** (`/`) - Overview with statistics and charts
2. **Results** (`/results`) - Complete results table
3. **Detail** (`/detail/:id`) - Individual analysis view
4. **Methodology** (`/methodology`) - System documentation

## 🔧 Configuration

### Environment Variables
```bash
# Results file location (default: /results/results.csv)
VITE_RESULTS_PATH=/path/to/results.csv

# Evidence file location (optional)
VITE_EVIDENCE_PATH=/path/to/evidence.json
```

### Customization
- Modify `src/utils/dataLoader.js` for different data formats
- Update colors in `tailwind.config.js`
- Adjust chart configurations in component files

## 🚦 Build & Deploy

```bash
# Development
npm run dev
# Opens http://localhost:3000

# Production Build
npm run build
# Output: ./dist/

# Preview Build
npm run preview
```

## 📋 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## ⚠️ Important Notes

### ❌ What This Interface Does NOT Do
- Modify ML models or predictions
- Recompute any analysis
- Store new data to databases
- Allow user input that affects results
- Claim accuracy or performance metrics

### ✅ What This Interface Does
- Display pre-computed results clearly
- Explain system methodology transparently
- Provide evidence for each decision
- Enable result exploration and analysis
- Demonstrate professional frontend development

## 🎯 Hackathon Goals

This frontend demonstrates:

1. **Technical Excellence**: Clean React architecture with modern tooling
2. **User Experience**: Intuitive, responsive design
3. **Transparency**: Clear explanation of ML decisions
4. **Professionalism**: Judge-ready presentation
5. **Safety**: Read-only interface respecting frozen backend

## 📄 License

Built for Kharagpur Data Science Hackathon 2026 - Track A.
