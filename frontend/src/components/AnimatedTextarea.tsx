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
  (
    {
      value,
      onChange,
      onKeyPress,
      onFocus,
      onBlur,
      placeholder,
      isLoading,
      disabled,
    },
    ref
  ) => {
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
      <div className="relative">
        {/* Animated border container */}
        <div className="relative rounded-2xl p-[1px]">
          {/* Animated gradient border */}
          <motion.div
            className="absolute inset-0 rounded-2xl"
            animate={{
              background: [
                "conic-gradient(from 0deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
                "conic-gradient(from 90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
                "conic-gradient(from 180deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
                "conic-gradient(from 270deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
                "conic-gradient(from 360deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57, #ff9ff3, #54a0ff, #ff6b6b)",
              ],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "linear",
            }}
            style={{
              opacity: isLoading ? 1 : 0,
            }}
          />

          {/* Inner container with background */}
          <div className="relative rounded-2xl overflow-hidden">
            {/* Background */}
            <div
              className="absolute inset-0 rounded-2xl pointer-events-none dark:hidden z-0"
              style={{
                background:
                  isFocused || isLoading
                    ? "rgba(255, 255, 255, 0.95)"
                    : "rgba(255, 255, 255, 0.8)",
              }}
            />
            {/* Dark mode background */}
            <div
              className="absolute inset-0 rounded-2xl pointer-events-none hidden dark:block z-0"
              style={{
                background:
                  isFocused || isLoading
                    ? "rgba(15, 23, 42, 0.95)"
                    : "rgba(15, 23, 42, 0.9)",
              }}
            />

            {/* Inner shadow effect - on top of background */}
            <div
              className="absolute inset-0 rounded-2xl pointer-events-none z-10"
              style={{
                boxShadow: isLoading
                  ? "inset 0 0 10px rgba(255, 107, 107, 0.3), inset 0 0 20px rgba(78, 205, 196, 0.25), inset 0 0 30px rgba(69, 183, 209, 0.2), inset 0 0 40px rgba(150, 206, 180, 0.15), inset 0 0 50px rgba(254, 202, 87, 0.1), inset 0 0 60px rgba(255, 159, 243, 0.08)"
                  : "none",
              }}
            />

            <textarea
              ref={ref}
              value={value}
              onChange={onChange}
              onKeyPress={onKeyPress}
              onFocus={handleFocus}
              onBlur={handleBlur}
              placeholder={placeholder}
              disabled={disabled}
              className={`w-full min-h-[52px] max-h-32 px-5 py-4 bg-transparent rounded-2xl resize-none transition-all duration-300 relative z-10 border-2 border-gray-300 dark:border-gray-600 ${
                disabled ? "cursor-not-allowed opacity-60" : "cursor-text"
              } ${
                isLoading
                  ? "text-gray-600 dark:text-gray-400"
                  : "text-gray-800 dark:text-gray-100"
              } placeholder-gray-500 dark:placeholder-gray-400 font-medium`}
              style={{
                fontFamily: "inherit",
                lineHeight: "1.5",
                outline: "none",
                boxShadow: "none",
              }}
            />
          </div>
        </div>

        {/* Smart indicator */}
        {/* <AnimatePresence>
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
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
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
        </AnimatePresence> */}
      </div>
    );
  }
);

AnimatedTextarea.displayName = "AnimatedTextarea";

export default AnimatedTextarea;
