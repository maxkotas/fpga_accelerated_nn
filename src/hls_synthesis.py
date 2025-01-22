# src/hls_synthesis.py
import os
import subprocess

def run_hls_synthesis(project_dir):
    if not os.path.exists(project_dir):
        print(f"Error: Project directory '{project_dir}' does not exist.")
        return

    tcl_script_path = os.path.join(project_dir, "project.tcl")
    if not os.path.exists(tcl_script_path):
        print(f"Error: TCL script not found in '{project_dir}'. Expected at: {tcl_script_path}")
        return

    print(f"Running HLS synthesis for project in: {project_dir}")

    try:
        subprocess.run(
            ["vivado_hls", "-f", tcl_script_path],
            cwd=project_dir,
            check=True
        )
        print("Vivado HLS synthesis completed successfully.")
    except FileNotFoundError:
        print("Error: Vivado HLS is not installed or not found in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Vivado HLS synthesis failed: {e}")

if __name__ == "__main__":
    hls_project_dir = os.path.join("..", "hls4ml_model_qkeras")
    run_hls_synthesis(hls_project_dir)