# Contributing to Cambium

Thank you for your interest in contributing to Cambium! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/cambium-lib/cambium.git
   cd cambium
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Ensure all tests pass
   - Update documentation if needed

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Format and lint**
   ```bash
   black cambium/ tests/
   isort cambium/ tests/
   mypy cambium/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for public functions
- Keep functions focused and small

## Testing

- Write unit tests for new functionality
- Aim for high test coverage
- Test edge cases and error conditions

## Documentation

- Update README.md if adding new features
- Add examples to the examples/ directory
- Update docstrings for API changes

## Commit Message Format

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

## Questions?

Feel free to open an issue or discussion for any questions!
