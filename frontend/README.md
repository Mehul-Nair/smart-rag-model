# Smart AI Agent Frontend

A beautiful and intuitive React frontend for the Smart AI Agent backend, featuring smooth animations and modern UI design.

## Features

- 🎨 Modern, responsive design with Tailwind CSS
- ✨ Smooth animations using Framer Motion
- 💬 Real-time chat interface with typing indicators
- 🔄 Animated textarea with blue border flow effect during AI processing
- 📱 Mobile-friendly design
- 🎯 Beautiful gradient backgrounds and glassmorphism effects

## Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **Lucide React** for icons
- **FastAPI** backend integration

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn
- Backend server running on port 8000

### Installation

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

The app will open at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

## Features in Detail

### Animated Textarea

- Blue border animation flows across the textarea when AI is processing
- Pulsing shadow effect during loading
- Smooth transitions between states

### Chat Interface

- Message bubbles with different styles for user and AI
- Timestamp display for each message
- Auto-scroll to latest message
- Typing indicators with animated dots

### Responsive Design

- Works seamlessly on desktop, tablet, and mobile
- Adaptive layout with proper spacing
- Touch-friendly interface elements

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000` and expects:

- **POST /chat** - Send messages and receive AI responses
- **Request body**: `{ message: string, session_id?: string }`
- **Response**: `{ response: string }`

## Customization

### Colors

Modify the primary color scheme in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    500: '#3b82f6', // Main blue color
    // ... other shades
  }
}
```

### Animations

Customize animations in `tailwind.config.js`:

```javascript
animation: {
  'border-flow': 'borderFlow 2s ease-in-out infinite',
  'typing': 'typing 1.5s ease-in-out infinite',
  // ... other animations
}
```

## Development

### Project Structure

```
src/
├── components/
│   ├── AnimatedTextarea.tsx    # Animated input component
│   ├── ChatMessage.tsx         # Individual message display
│   └── TypingIndicator.tsx     # Loading animation
├── App.tsx                     # Main application component
├── index.tsx                   # Application entry point
└── index.css                   # Global styles and Tailwind imports
```

### Adding New Features

1. Create new components in the `components/` directory
2. Import and use them in `App.tsx`
3. Add any new styles to `index.css` or component files
4. Update TypeScript interfaces as needed

## Troubleshooting

### Common Issues

1. **Backend Connection Error**: Ensure the FastAPI server is running on port 8000
2. **Animation Not Working**: Check that Framer Motion is properly installed
3. **Styling Issues**: Verify Tailwind CSS is configured correctly

### Performance Tips

- Use React.memo() for components that don't need frequent re-renders
- Optimize animations by using `transform` and `opacity` properties
- Consider code splitting for larger applications

## License

This project is part of the Smart AI Agent system.
