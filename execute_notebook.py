import nbformat as nbf
import io
import sys
import base64
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)

    # Change directory to notebook folder so relative paths work
    nb_dir = os.path.dirname(nb_path)
    old_cwd = os.getcwd()
    if nb_dir:
        os.chdir(nb_dir)

    global_env = {}

    exec_count = 1
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            # Strip IPython magic commands like %matplotlib inline for exec
            exec_source = "\n".join([line for line in source.split("\n") if not line.strip().startswith("%")])
            
            # Capture stdout
            buffer = io.StringIO()
            sys.stdout = buffer
            
            # Clear matplotlib figures before running
            plt.close('all')
            
            outputs = []
            try:
                exec(exec_source, global_env)
                
                # Check for printed text
                stdout_text = buffer.getvalue()
                if stdout_text:
                    outputs.append(nbf.v4.new_output(
                        output_type='stream',
                        name='stdout',
                        text=stdout_text
                    ))
                
                # Check for matplotlib figures generated
                figs = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figs:
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
                    img_buf.seek(0)
                    img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
                    
                    outputs.append(nbf.v4.new_output(
                        output_type='display_data',
                        data={'image/png': img_b64, 'text/plain': '<Figure size ...>'},
                        metadata={}
                    ))
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Error in cell execution: {e}")
                outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stderr',
                    text=str(e)
                ))
            finally:
                sys.stdout = sys.__stdout__
            
            cell.outputs = outputs
            cell.execution_count = exec_count
            exec_count += 1

    os.chdir(old_cwd)
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    print(f"Successfully executed all cells in '{nb_path}' and saved outputs!")

if __name__ == "__main__":
    run_notebook(os.path.join("Superstore_store_dataset", "Superstore_Descriptive_Analytics.ipynb"))
