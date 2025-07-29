# Contributing to Llama JAX Implementation

Thank you for your interest in contributing to the Llama JAX Implementation! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Check existing issues**: Before creating a new issue, please check if a similar issue already exists.
2. **Use the issue template**: When creating an issue, please use the provided template and include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, JAX version)
   - Error messages and stack traces

### Suggesting Enhancements

1. **Feature requests**: Use the feature request template
2. **Be specific**: Describe the enhancement clearly and explain why it would be useful
3. **Consider implementation**: If possible, suggest how the feature could be implemented

### Code Contributions

#### Getting Started

1. **Fork the repository**: Click the "Fork" button on GitHub
2. **Clone your fork**: `git clone https://github.com/yourusername/llama-jax.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Set up environment**: Follow the installation instructions in README.md

#### Development Guidelines

1. **Code Style**: Follow PEP 8 guidelines
2. **Documentation**: Add docstrings to new functions and classes
3. **Tests**: Write tests for new functionality
4. **Type Hints**: Use type hints where appropriate
5. **Comments**: Add comments for complex logic

#### Testing

Before submitting a pull request:

1. **Run existing tests**: `python -m pytest tests/`
2. **Add new tests**: Create tests for new functionality
3. **Check code style**: Use tools like `black` and `flake8`
4. **Test on different environments**: If possible, test on different Python versions

#### Submitting Changes

1. **Commit your changes**: Use clear, descriptive commit messages
2. **Push to your fork**: `git push origin feature/your-feature-name`
3. **Create a pull request**: Use the PR template and describe your changes
4. **Wait for review**: Maintainers will review your code and provide feedback

## 📋 Development Setup

### Prerequisites

- Python 3.8+
- JAX (CPU/GPU support)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/llama-jax.git
cd llama-jax

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=llama_jax

# Run specific test file
python -m pytest tests/test_llama_jax.py
```

### Code Quality

```bash
# Format code
black llama_jax.py examples/ tests/

# Check code style
flake8 llama_jax.py examples/ tests/

# Type checking
mypy llama_jax.py examples/ tests/
```

## 🏗️ Project Structure

```
llama-jax/
├── llama_jax.py              # Main implementation
├── examples/                  # Example scripts
├── tests/                    # Test files
├── docs/                     # Documentation
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # Project documentation
```

## 📝 Code Style Guide

### Python Code

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add type hints where appropriate
- Keep functions focused and small
- Use descriptive docstrings

### Documentation

- Use clear, concise language
- Include code examples
- Update documentation when changing functionality
- Use markdown for formatting

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment**: OS, Python version, JAX version
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full error messages and stack traces
6. **Code example**: Minimal code to reproduce the issue

## 💡 Feature Requests

When suggesting features:

1. **Clear description**: What the feature should do
2. **Use case**: Why this feature would be useful
3. **Implementation ideas**: How it could be implemented
4. **Priority**: How important this feature is

## 📞 Getting Help

If you need help:

1. **Check documentation**: Read the README and docs
2. **Search issues**: Look for similar questions
3. **Create an issue**: Use the appropriate template
4. **Join discussions**: Participate in GitHub discussions

## 🙏 Recognition

Contributors will be recognized in:

- The README file
- Release notes
- GitHub contributors page

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Llama JAX Implementation! 🚀 