import os
import subprocess
import sys


def run_command(command, error_message):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"\n❌ {error_message}")
        sys.exit(1)


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def main():
    print("=== Python ML Project Setup ===\n")

    user_name = input("Enter environment name suffix (default: myname): ").strip()
    if not user_name:
        user_name = "myname"

    env_name = f"venv-{user_name}"

    print(f"\nEnvironment folder will be: {env_name}")

    # 1) Create virtual environment
    if os.path.exists(env_name):
        print(f"⚠️ Folder '{env_name}' already exists.")
        answer = input("Do you want to reuse it? (y/n): ").strip().lower()
        if answer != "y":
            print("Setup cancelled.")
            sys.exit(0)
    else:
        print(f"Creating virtual environment: {env_name}")
        run_command(
            [sys.executable, "-m", "venv", env_name],
            "Failed to create the virtual environment."
        )

    # 2) Define paths
    if os.name == "nt":
        python_path = os.path.join(env_name, "Scripts", "python.exe")
        pip_path = os.path.join(env_name, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(env_name, "bin", "python")
        pip_path = os.path.join(env_name, "bin", "pip")

    if not os.path.exists(python_path):
        print("\n❌ Python executable was not found inside the environment.")
        sys.exit(1)

    # 3) Upgrade pip
    print("Upgrading pip...")
    run_command(
        [python_path, "-m", "pip", "install", "--upgrade", "pip"],
        "Failed to upgrade pip."
    )

    # 4) Install requirements
    if os.path.exists("requirements.txt"):
        print("Installing packages from requirements.txt ...")
        run_command(
            [pip_path, "install", "-r", "requirements.txt"],
            "Failed to install packages from requirements.txt."
        )
    else:
        print("No requirements.txt found. Skipping package installation.")

    # 5) Install ipykernel
    print("Installing ipykernel...")
    run_command(
        [pip_path, "install", "ipykernel"],
        "Failed to install ipykernel."
    )

    # 6) Register Jupyter kernel
    print("Registering Jupyter kernel...")
    run_command(
        [
            python_path,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            env_name,
            "--display-name",
            f"Python ({env_name})",
        ],
        "Failed to register the Jupyter kernel."
    )

    # 7) Create project structure beside the environment
    print("\nCreating project folders...")
    project_folders = [
        "data",
        "notebooks",
        "src",
        "models",
        "outputs"
    ]

    for folder in project_folders:
        create_folder(folder)

    # Optional starter files
    starter_files = [
        os.path.join("src", "__init__.py"),
        os.path.join("src", "train.py"),
        os.path.join("src", "predict.py"),
        "README.md",
        ".gitignore"
    ]

    for file_path in starter_files:
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                if file_path == ".gitignore":
                    f.write(f"{env_name}/\n__pycache__/\n.ipynb_checkpoints/\n*.pyc\n")
            print(f"Created file: {file_path}")
        else:
            print(f"File already exists: {file_path}")

    print("\n✅ Environment setup completed successfully!")
    print(f"Environment name : {env_name}")
    print(f"Interpreter path : {python_path}")

    print("\nProject structure created:")
    print(f"""
Exercise/
│
├── {env_name}/
│   ├── Include
│   ├── Lib
│   └── Scripts
│
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
""")

    print("Next steps in VS Code:")
    print("1. Open the project folder")
    print("2. Press Ctrl+Shift+P")
    print("3. Select: Python: Select Interpreter")
    print(f"4. Choose: {python_path}")
    print("5. Open your notebook and select the new kernel")


if __name__ == "__main__":
    main()