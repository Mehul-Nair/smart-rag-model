import React, { forwardRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface AnimatedTextareaProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onFocus: () => void;
  onBlur: () => void;
  placeholder: string;
  isLoading: boolean;
  disabled: boolean;
}

const AnimatedTextarea = forwardRef<HTMLTextAreaElement, AnimatedTextareaProps>(
  ({ value, onChange, onKeyPress, onFocus, onBlur, placeholder, isLoading, disabled }, ref) => {
    const [isFocused, setIsFocused] = React.useState(false);

    const handleFocus = () => {
      setIsFocused(true);
      onFocus();
    };

    const handleBlur = () => {
      setIsFocused(false);
      onBlur();
    };

    return (
      <motion.div 
        className="relative"
        animate={{ 
          scale: isFocused ? 1.02 : 1,
        }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <div className="relative">
          {/* Gradient border container */}
          <motion.div
            className="absolute inset-0 rounded-2xl p-[2px] overflow-hidden"
            animate={{
              background: (isFocused || isLoading) ? [
                "linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff)",
                "linear-gradient(90deg, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
                "linear-gradient(135deg, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b, #4ecdc4)",
                "linear-gradient(180deg, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b, #4ecdc4, #45b7d1)",
                "linear-gradient(225deg, #feca57, #ff9ff3, #54a0ff, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4)",
                "linear-gradient(270deg, #ff9ff3, #54a0ff, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57)",
                "linear-gradient(315deg, #54a0ff, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3)",
                "linear-gradient(360deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff)"
              ] : "transparent"
            }}
            transition={{
              duration: (isFocused || isLoading) ? 3 : 0.3,
              repeat: (isFocused || isLoading) ? Infinity : 0,
              ease: "linear",
            }}
          >
            <div className="w-full h-full bg-white rounded-2xl" />
          </motion.div>

          {/* Main container */}
          <motion.div
            className="relative rounded-2xl overflow-hidden"
            animate={{
              boxShadow: isLoading 
                ? [
                    "0 0 0 0 rgba(255, 107, 107, 0.4)",
                    "0 0 0 8px rgba(255, 107, 107, 0.1)",
                    "0 0 0 0 rgba(255, 107, 107, 0)",
                  ]
                : isFocused
                ? "0 8px 32px rgba(255, 107, 107, 0.15)"
                : "0 4px 16px rgba(0, 0, 0, 0.1)"
            }}
            transition={{
              duration: isLoading ? 2 : 0.3,
              repeat: isLoading ? Infinity : 0,
              ease: "easeInOut",
            }}
          >
            {/* Background gradient */}
            <div
              className="absolute inset-0 rounded-2xl pointer-events-none"
              style={{
                background:
                  isFocused || isLoading
                    ? "linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.9) 100%)"
                    : "linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.4) 100%)",
              }}
            />
            {/* Animated shimmer effect for loading */}
            {isLoading && (
              <motion.div
                className="absolute inset-0 rounded-2xl pointer-events-none"
                animate={{ 
                  background: [
                    "linear-gradient(90deg, transparent 0%, rgba(255, 107, 107, 0.3) 50%, transparent 100%)",
                    "linear-gradient(90deg, transparent 0%, rgba(255, 107, 107, 0.3) 50%, transparent 100%)"
                  ],
                  x: ["-100%", "100%"]
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  ease: "linear",
                }}
                style={{
                  background: "linear-gradient(90deg, transparent 0%, rgba(255, 107, 107, 0.3) 50%, transparent 100%)",
                  width: "100%",
                  height: "100%",
                }}
              />
            )}

            <textarea
              ref={ref}
              value={value}
              onChange={onChange}
              onKeyPress={onKeyPress}
              onFocus={handleFocus}
              onBlur={handleBlur}
              placeholder={placeholder}
              disabled={disabled}
              className={`w-full min-h-[52px] max-h-32 px-5 py-4 bg-transparent backdrop-blur-sm rounded-2xl resize-none outline-none border-none focus:outline-none focus:ring-0 focus:border-none transition-all duration-300 relative z-10 ${
                disabled ? "cursor-not-allowed opacity-60" : "cursor-text"
              } ${
                isLoading ? "text-gray-600" : "text-gray-800"
              } placeholder-gray-500 font-medium`}
              style={{
                fontFamily: "inherit",
                lineHeight: "1.5",
                boxShadow: "none",
              }}
            />
          </motion.div>
        </div>

        {/* Smart indicator */}
        <AnimatePresence>
          {(isLoading || isFocused) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.8, y: 10 }}
              className="absolute -top-3 -right-3 flex items-center space-x-2"
            >
              {isLoading ? (
                <div className="w-6 h-6 bg-gradient-to-br from-pink-500 to-orange-500 rounded-full flex items-center justify-center shadow-lg">
                  <motion.div
                    className="w-3 h-3 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              ) : (
                <motion.div
                  className="w-6 h-6 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center shadow-lg"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <div className="w-2 h-2 bg-white rounded-full" />
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    );
  }
);

AnimatedTextarea.displayName = "AnimatedTextarea";

export default AnimatedTextarea;
