# Typing TUI

A terminal-based typing practice application built with Python and Textual. Improve your typing speed and accuracy with timed practice sessions, multiple modes, difficulty levels, and customizable themes.

## Features

- ⏱️ **Timed Practice Sessions**: Default 60-second tests (configurable)
- 📊 **Real-time Statistics**: Track WPM (Words Per Minute), CWPM (Correct WPM), and accuracy
- 🎯 **Multiple Modes**:
  - **Words**: Common English words
  - **Quotes**: Inspirational quotes
  - **Code Tokens**: Programming keywords, symbols, and syntax
  - **Hacker**: Kali Linux tools and security tooling names
- 🎚️ **Difficulty Levels**: Easy, Normal, Hard
- 🎨 **Customizable Themes**: Built-in themes (slate, ember, mint) with support for custom themes
- 📈 **Score Tracking**: Local-only score history with top scores display
- ⌨️ **Keyboard Shortcuts**: Quick access to all features
- 🖥️ **Terminal UI**: Beautiful, responsive TUI built with Textual

## Requirements

- Python 3.7+
- `rich >= 13.7.0`
- `textual >= 0.65.0`

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install rich textual
```

## Usage

Run the application:

```bash
python typing_tui.py
```

### Keyboard Shortcuts

- `Ctrl+R` - Restart the current session
- `Ctrl+N` - Cycle through modes (words → quotes → code tokens → hacker)
- `Ctrl+D` - Cycle through difficulty levels (easy → normal → hard)
- `Ctrl+T` - Cycle through themes (slate → ember → mint → ...)
- `Ctrl+Q` - Quit the application

### Typing

- Start typing when the session begins
- Use space to advance to the next word
- The current word is highlighted with underline
- Correct characters show in green, incorrect in red
- Errors flash the prompt area briefly

## Configuration

Create a `typing_tui.config.json` file in the same directory as `typing_tui.py` to customize defaults:

```json
{
  "theme": "slate",
  "mode": "words",
  "difficulty": "normal",
  "duration_sec": 60,
  "themes": {
    "custom_theme": {
      "screen_bg": "#000000",
      "card_bg": "#111111",
      "title": "#ffffff",
      ...
    }
  }
}
```

### Configuration Options

- `theme`: Default theme name (default: "slate")
- `mode`: Default mode - "words", "quotes", "code tokens", or "hacker" (default: "words")
- `difficulty`: Default difficulty - "easy", "normal", or "hard" (default: "normal")
- `duration_sec`: Session duration in seconds (default: 60)
- `themes`: Object with custom theme definitions (see code for available theme keys)

## Score Storage

Scores are stored locally in:
- **macOS**: `~/Library/Application Support/typing-tui/scores.json`
- **Linux**: `$XDG_DATA_HOME/typing-tui/scores.json` or `~/.local/share/typing-tui/scores.json`

The application keeps the top 50 scores, sorted by correct WPM (CWPM) and accuracy.

## Statistics

Each session tracks:
- **WPM**: Words Per Minute (based on total characters typed)
- **CWPM**: Correct Words Per Minute (based on correctly typed characters)
- **Accuracy**: Percentage of correct characters
- **Words Correct/Wrong**: Count of correctly and incorrectly completed words
- **Time Remaining**: Countdown timer with progress bar

After each session, you'll see:
- Last run summary
- Best score
- Top 3 scores
- Words with most mistakes
- Slowest words

## Modes Explained

### Words Mode
Practice with common English words from the built-in word list. Difficulty affects word length:
- **Easy**: 3-6 characters
- **Normal**: 2-8 characters
- **Hard**: 2-10 characters + extra tokens

### Quotes Mode
Type inspirational programming and life quotes. Words are selected from a curated list of quotes.

### Code Tokens Mode
Practice typing programming-related tokens:
- **Easy**: Alphanumeric tokens only (e.g., "def", "class", "True")
- **Normal**: Tokens up to 8 characters (e.g., "print()", "range()")
- **Hard**: All tokens including symbols (e.g., "==", "{}", "snake_case", "0xFF")

### Hacker Mode
Type names of security tools and Kali Linux utilities:
- **Easy**: 3-8 characters
- **Normal**: 3-12 characters
- **Hard**: 2-18 characters

## Themes

Built-in themes:
- **slate**: Dark blue-gray theme (default)
- **ember**: Warm orange/amber theme
- **mint**: Cool green/teal theme

Custom themes can be added via the configuration file.

## License

This project is open source. Feel free to modify and distribute as needed.

## Contributing

Contributions are welcome! Feel free to:
- Add new word pools or quote collections
- Create new themes
- Improve the UI/UX
- Add new modes or features
- Report bugs or suggest improvements
