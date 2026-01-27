# Depth Mapping Project

A project combining Python and C++ for depth mapping functionality.

## Setup

### Python Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run Python code:
   ```bash
   python main.py
   ```

### C++ Setup

**Note:** You need a C++ compiler installed. On Windows, you can use:
- **Visual Studio** (includes MSVC compiler) - Recommended
- **MinGW-w64** (provides g++)
- **Clang** (via LLVM)

#### Using CMake with Visual Studio (Windows)

1. Install Visual Studio with "Desktop development with C++" workload
2. Open "Developer Command Prompt for VS" or run:
   ```powershell
   & "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   ```
3. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

4. Generate build files:
   ```bash
   cmake .. -G "Visual Studio 17 2022" -A x64
   ```

5. Build the project:
   ```bash
   cmake --build . --config Release
   ```

6. Run the executable:
   ```bash
   .\Release\depth_mapping.exe
   ```

#### Using CMake with MinGW (Windows)

1. Install MinGW-w64 and add to PATH
2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Generate build files:
   ```bash
   cmake .. -G "MinGW Makefiles"
   ```

4. Build the project:
   ```bash
   cmake --build .
   ```

5. Run the executable:
   ```bash
   .\depth_mapping.exe
   ```

#### Using g++ directly (Alternative)

```bash
g++ -std=c++17 -o depth_mapping main.cpp
.\depth_mapping.exe
```

## Project Structure

```
depth_mapping/
├── main.py              # Python entry point
├── main.cpp             # C++ entry point
├── requirements.txt     # Python dependencies
├── CMakeLists.txt       # CMake build configuration
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Development

- Add Python modules in separate `.py` files
- Add C++ source files and update `CMakeLists.txt` accordingly
- Update `requirements.txt` with Python dependencies as needed

