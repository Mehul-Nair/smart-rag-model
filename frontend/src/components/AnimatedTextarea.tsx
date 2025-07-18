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
        <motion.div
          className={`relative rounded-2xl border-2 transition-all duration-500 overflow-hidden ${
            isLoading
              ? "border-blue-400 shadow-xl shadow-blue-500/30"
              : isFocused
              ? "border-blue-300 shadow-lg shadow-blue-500/20"
              : "border-white/40 shadow-md"
          }`}
          animate={{
            boxShadow: isLoading 
              ? [
                  "0 0 0 0 rgba(59, 130, 246, 0.4)",
                  "0 0 0 8px rgba(59, 130, 246, 0.1)",
                  "0 0 0 0 rgba(59, 130, 246, 0)",
                ]
              : isFocused
              ? "0 8px 32px rgba(59, 130, 246, 0.15)"
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
                  ? "linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.05) 100%)"
                  : "linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.4) 100%)",
            }}
          />

          {/* Animated shimmer effect for loading */}
          {isLoading && (
            <motion.div
              className="absolute inset-0 rounded-2xl pointer-events-none"
              animate={{ 
                background: [
                  "linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.3) 50%, transparent 100%)",
                  "linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.3) 50%, transparent 100%)"
                ],
                x: ["-100%", "100%"]
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "linear",
              }}
              style={{
                background: "linear-gradient(90deg, transparent 0%, rgba(59, 130, 246, 0.3) 50%, transparent 100%)",
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
            className={`w-full min-h-[52px] max-h-32 px-5 py-4 bg-transparent backdrop-blur-sm rounded-2xl resize-none outline-none transition-all duration-300 relative z-10 ${
              disabled ? "cursor-not-allowed opacity-60" : "cursor-text"
            } ${
              isLoading ? "text-gray-600" : "text-gray-800"
            } placeholder-gray-500 font-medium`}
            style={{
              fontFamily: "inherit",
              lineHeight: "1.5",
            }}
          />
        </motion.div>

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
                <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
                  <motion.div
                    className="w-3 h-3 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  />
                </div>
              ) : (
                <motion.div
                  className="w-6 h-6 bg-gradient-to-br from-green-400 to-blue-500 rounded-full flex items-center justify-center shadow-lg"
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
