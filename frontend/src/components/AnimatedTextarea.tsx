import React, { forwardRef } from "react";
import { motion } from "framer-motion";

interface AnimatedTextareaProps {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  placeholder: string;
  isLoading: boolean;
  disabled: boolean;
}

const AnimatedTextarea = forwardRef<HTMLTextAreaElement, AnimatedTextareaProps>(
  ({ value, onChange, onKeyPress, placeholder, isLoading, disabled }, ref) => {
    return (
      <div className="relative">
        <motion.div
          className={`relative rounded-xl border-2 transition-all duration-300 ${
            isLoading
              ? "border-primary-500 shadow-lg shadow-primary-500/20"
              : "border-gray-200"
          }`}
          animate={
            isLoading
              ? {
                  boxShadow: [
                    "0 0 0 0 rgba(59, 130, 246, 0.4)",
                    "0 0 0 10px rgba(59, 130, 246, 0)",
                    "0 0 0 0 rgba(59, 130, 246, 0)",
                  ],
                }
              : {}
          }
          transition={
            isLoading
              ? {
                  duration: 1.5,
                  repeat: Infinity,
                  ease: "easeInOut",
                }
              : {}
          }
        >
          {/* Default blue gradient in top-left corner */}
          <div
            className="absolute inset-0 rounded-xl pointer-events-none overflow-hidden"
            style={{
              background:
                "linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, transparent 30%)",
              backgroundSize: "100% 100%",
              backgroundPosition: "0 0",
              backgroundRepeat: "no-repeat",
            }}
          />

          {/* Animated border overlay for loading state */}
          {isLoading && (
            <motion.div
              className="absolute inset-0 rounded-xl pointer-events-none overflow-hidden"
              style={{
                background:
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 0%, transparent 0%)",
                backgroundSize: "100% 2px",
                backgroundPosition: "0 0",
                backgroundRepeat: "no-repeat",
              }}
              animate={{
                background: [
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 0%, transparent 0%)",
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 25%, transparent 25%)",
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 50%, transparent 50%)",
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 75%, transparent 75%)",
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 100%, transparent 100%)",
                  "linear-gradient(90deg, rgba(59, 130, 246, 0.6) 100%, transparent 100%)",
                ],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          )}

          <textarea
            ref={ref}
            value={value}
            onChange={onChange}
            onKeyPress={onKeyPress}
            placeholder={placeholder}
            disabled={disabled}
            className={`w-full min-h-[48px] max-h-32 px-4 py-3 bg-white/80 backdrop-blur-sm rounded-xl resize-none outline-none transition-all duration-200 ${
              disabled ? "cursor-not-allowed opacity-60" : "cursor-text"
            } ${
              isLoading ? "text-gray-600" : "text-gray-900"
            } placeholder-gray-400`}
            style={{
              fontFamily: "inherit",
              lineHeight: "1.5",
            }}
          />
        </motion.div>

        {/* Loading indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute -top-2 -right-2 w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center"
          >
            <motion.div
              className="w-3 h-3 border-2 border-white border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
          </motion.div>
        )}
      </div>
    );
  }
);

AnimatedTextarea.displayName = "AnimatedTextarea";

export default AnimatedTextarea;
