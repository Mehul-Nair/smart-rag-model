import React from "react";
import { motion } from "framer-motion";

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-center space-x-2">
      <span className="text-xs text-gray-500 dark:text-gray-400 font-medium mr-2">
        Thinking
      </span>
      <motion.div
        className="w-2 h-2 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full"
        animate={{
          scale: [1, 1.5, 1],
          opacity: [0.4, 1, 0.4],
        }}
        transition={{
          duration: 1.2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0,
        }}
      />
      <motion.div
        className="w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full"
        animate={{
          scale: [1, 1.5, 1],
          opacity: [0.4, 1, 0.4],
        }}
        transition={{
          duration: 1.2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.2,
        }}
      />
      <motion.div
        className="w-2 h-2 bg-gradient-to-r from-pink-400 to-red-500 rounded-full"
        animate={{
          scale: [1, 1.5, 1],
          opacity: [0.4, 1, 0.4],
        }}
        transition={{
          duration: 1.2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.4,
        }}
      />
    </div>
  );
};

export default TypingIndicator;
