# OpenAI API Key Setup Guide

This guide will help you fix OpenAI API key issues and set up proper environment variable management.

## 🚨 Problem Description

You're experiencing issues where:

- Your old OpenAI API key is still being used from system environment variables
- You need to explicitly set the new key for every new terminal instance
- Some parts of the codebase use environment variables while others use `.env` files

## 🔧 Solution Overview

We've implemented a comprehensive solution that:

1. **Centralizes environment variable management** in `config.py`
2. **Provides validation functions** to check API key status
3. **Creates setup scripts** to manage environment variables easily
4. **Ensures consistent loading** across all modules

## 📋 Step-by-Step Fix

### Step 1: Check Current Status

First, let's see what's currently happening:

```bash
cd backend
python setup_env.py
```

Choose option 2 to check current environment status.

### Step 2: Clear Old System Environment Variable

**Windows:**

```cmd
# Option 1: Use the batch script
setup_env.bat

# Option 2: Manual command
set OPENAI_API_KEY=
```

**Linux/Mac:**

```bash
unset OPENAI_API_KEY
```

**Note:** You may need to restart your terminal for changes to take effect.

### Step 3: Create .env File

```bash
cd backend
python setup_env.py
```

Choose option 1 to create/update the `.env` file.

### Step 4: Update Your API Key

Edit the `.env` file in the backend directory:

```env
# Replace this line:
OPENAI_API_KEY=your_openai_api_key_here

# With your actual API key:
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 5: Verify Configuration

```bash
python -c "from config import validate_openai_key; print('✅ Valid' if validate_openai_key() else '❌ Invalid')"
```

## 🔄 How It Works

### Environment Variable Loading Order

The system now follows this priority order:

1. **`.env` file** (highest priority)
2. **System environment variables** (fallback)
3. **Default values** (if configured)

### Validation Functions

- `validate_openai_key()`: Checks if API key is properly set
- `get_openai_key()`: Returns validated API key or raises error

### Files Updated

- `config.py`: Added validation and centralized loading
- `main.py`: Uses new validation functions
- `build_index.py`: Uses new validation functions
- `setup_env.py`: New setup script
- `setup_env.bat`: Windows batch script

## 🛠️ Usage

### For Development

```bash
# Start with proper environment
cd backend
python setup_env.py  # Set up .env file
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### For Production

```bash
# Set system environment variable
export OPENAI_API_KEY=your-api-key  # Linux/Mac
set OPENAI_API_KEY=your-api-key     # Windows

# Or use .env file (recommended)
# Create .env file in backend directory
```

## 🔍 Troubleshooting

### Issue: Still seeing old API key

**Solution:**

1. Restart your terminal/command prompt
2. Check if the key is set in system environment variables
3. Use `setup_env.bat` (Windows) or `unset OPENAI_API_KEY` (Linux/Mac)

### Issue: "Missing OpenAI API Key" error

**Solution:**

1. Make sure `.env` file exists in backend directory
2. Verify the API key is not set to placeholder value
3. Run `python setup_env.py` to check status

### Issue: Permission denied on Windows

**Solution:**

1. Run command prompt as administrator
2. Or use the `.env` file approach instead

### Issue: Key works in one terminal but not another

**Solution:**

1. Use `.env` file approach (recommended)
2. Or set the environment variable in each terminal session

## 📝 Best Practices

1. **Use `.env` files** for development (easier to manage)
2. **Use system environment variables** for production (more secure)
3. **Never commit API keys** to version control
4. **Validate configuration** before starting the application
5. **Use the setup scripts** to manage environment variables

## 🔐 Security Notes

- The `.env` file is already in `.gitignore`
- API keys are never logged or displayed in full
- Validation functions check for placeholder values
- System environment variables take precedence over `.env` files

## 📞 Support

If you're still having issues:

1. Run `python setup_env.py` and check the output
2. Verify your API key is valid
3. Check that `.env` file exists and is properly formatted
4. Restart your terminal/IDE after making changes
