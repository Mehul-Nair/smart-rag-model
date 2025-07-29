import React from "react";
import { motion } from "framer-motion";
import { Bot, Shield, Clock, CheckCircle } from "lucide-react";

interface WarrantyResponseData {
  type: string;
  message: string;
}

interface WarrantyResponseProps {
  data: WarrantyResponseData;
  timestamp: Date;
}

const WarrantyResponse: React.FC<WarrantyResponseProps> = ({
  data,
  timestamp,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="w-full"
    >
      <div className="flex items-start space-x-3 chat-message-container">
        {/* Avatar */}
        <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-green-500 rounded-full flex items-center justify-center flex-shrink-0">
          <Shield className="w-4 h-4 text-white" />
        </div>

        {/* Content */}
        <div className="flex-1 flex-stable">
          <div className="inline-block px-5 py-4 rounded-2xl shadow-lg max-w-xs lg:max-w-md backdrop-blur-sm border bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 text-gray-800 dark:text-gray-100 border-green-200/50 dark:border-green-600/50">
            {/* Warranty Header */}
            <div className="flex items-center space-x-2 mb-3">
              <Shield className="w-5 h-5 text-green-600 dark:text-green-400" />
              <span className="font-semibold text-green-700 dark:text-green-300 text-sm">
                Warranty Information
              </span>
            </div>

            {/* Warranty Content */}
            <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium text-gray-700 dark:text-gray-200">
              {data.message}
            </div>

            {/* Warranty Features */}
            <div className="mt-3 pt-3 border-t border-green-200/50 dark:border-green-600/50">
              <div className="flex items-center space-x-2 text-xs text-green-600 dark:text-green-400">
                <CheckCircle className="w-3 h-3" />
                <span>Manufacturing defects covered</span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-green-600 dark:text-green-400 mt-1">
                <Clock className="w-3 h-3" />
                <span>Standard warranty period applies</span>
              </div>
            </div>
          </div>

          {/* Timestamp */}
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-1">
            {timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default WarrantyResponse;
