import React from "react";
import { motion } from "framer-motion";
import {
  Bot,
  Package,
  Tag,
  Palette,
  Ruler,
  Star,
  ShoppingCart,
} from "lucide-react";

interface ProductDetailData {
  type: string;
  product_name: string;
  details: Record<string, string>;
  message: string;
}

interface ProductDetailResponseProps {
  data: ProductDetailData;
  timestamp: Date;
}

const ProductDetailResponse: React.FC<ProductDetailResponseProps> = ({
  data,
  timestamp,
}) => {
  const formatDetailValue = (key: string, value: string): string => {
    if (key === "price") {
      return `₹${value}`;
    }
    return value;
  };

  const getDetailIcon = (key: string) => {
    switch (key.toLowerCase()) {
      case "brand":
        return <Star className="w-4 h-4" />;
      case "price":
        return <Tag className="w-4 h-4" />;
      case "material":
        return <Package className="w-4 h-4" />;
      case "color":
        return <Palette className="w-4 h-4" />;
      case "size":
        return <Ruler className="w-4 h-4" />;
      case "category":
        return <ShoppingCart className="w-4 h-4" />;
      default:
        return <Package className="w-4 h-4" />;
    }
  };

  const getDetailLabel = (key: string): string => {
    switch (key.toLowerCase()) {
      case "title":
        return "Product Name";
      case "brand":
        return "Brand";
      case "price":
        return "Price";
      case "material":
        return "Material";
      case "color":
        return "Color";
      case "size":
        return "Size";
      case "category":
        return "Category";
      default:
        return key.charAt(0).toUpperCase() + key.slice(1);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="flex items-start space-x-3 chat-message-container"
    >
      {/* Avatar */}
      <div className="w-8 h-8 bg-gradient-to-r from-primary-400 to-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-white" />
      </div>

      {/* Content */}
      <div className="flex-1 flex-stable">
        {/* Product Details Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 px-6 py-4 border-b border-gray-100 dark:border-gray-700">
            <div className="flex items-center space-x-2 mb-2">
              <Package className="w-5 h-5 text-primary-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Product Details
              </h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {data.product_name}
            </p>
          </div>

          {/* Details Grid */}
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {Object.entries(data.details).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                    {getDetailIcon(key)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                      {getDetailLabel(key)}
                    </p>
                    <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {formatDetailValue(key, value)}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            {/* Description */}
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-700/50">
              <div className="flex items-start space-x-2">
                <div className="flex-shrink-0 w-5 h-5 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center mt-0.5">
                  <Package className="w-3 h-3 text-blue-600 dark:text-blue-400" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">
                    Product Description
                  </p>
                  <p className="text-sm text-blue-800 dark:text-blue-200 leading-relaxed">
                    {data.message}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Timestamp */}
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 ml-1">
          {timestamp.toLocaleTimeString()}
        </div>
      </div>
    </motion.div>
  );
};

export default ProductDetailResponse;
