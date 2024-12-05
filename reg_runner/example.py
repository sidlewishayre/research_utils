import sys

# NOTE: To create your own version, copy and paste this code into your own directory
# and change the "save_folder" filepath.

# Also if in running the code, you get stuck at a breakpoint, type c + enter to continue and q + enter to quit.

# specify location of regression framework and import it
from reg_runner import run_tables

# specify where you want to save regression results
save_folder = None

# specifiy which data you want to use to run regressions on
data_filepath = None

# specify what extra stata code you want to run before regression
stata_code = """
gen income_ln = ln(income)
"""
regs = [
    {
        "id": "electric_ir",  # unique name identifying regression
        "reg_code": "reghdfe originalinterestrate electric hybrid if originalinterestrate > 0 & originalinterestrate<., absorb(state) vce(cluster loan_id)",
        # code to run regression
        "data": data_filepath,  # filepath of data on which to run regression
        "table": "interest_rate_by_engine",  # table in which to display regression results
        "stata_code": stata_code,  # code to run on filepath before running regression
        "subset": "No ZIRP",  # How to label subset. (Code automatically formats this for easy conditions, but this particular ondition is not supported yet.)
        "reload": False,  # whether or not to reload regression coefficients
    },
    {
        "id": "hybrid_ir",
        "reg_code": "reghdfe originalinterestrate electric hybrid income_ln if originalinterestrate > 0 & originalinterestrate<., absorb(state) vce(cluster loan_id)",
        "data": data_filepath,
        "table": "interest_rate_by_engine",
        "subset": "No ZIRP",
        "stata_code": stata_code,
    },
]


tables = [
    {
        "id": "interest_rate_by_engine",  # id of table which determines which regressions to include
        "title": "Interest Rate",  # title of table
        "index_vars": [
            "electric",
            "hybrid",
            "incomeln",  # notice not "income_ln" becuase underscore is removed in my STATA code, need to fix this.
            "cons",
        ],  # variables to include in rows
        "var_map": {
            "electric": "EV",
            "hybrid": "Hybrid",
            "cons": "Constant",
            "state": "State",
            "incomeln": "Income",
            "originalinterestrate": "Interest Rate",
        },  # dictionary renaming variables
    }
]

reg_table = run_tables(
    regs=regs,
    tables=tables,
    save_folder=save_folder,
    reg_kwargs={"silent": False},  # show stata output in console
    prevent_reload=False,  # do not rerun regressions even if they have changed
    reload=False,  # reload all regressions
    debug=False,  # only run on start of data set and do not save results
    overwrite=True,  # allow existing coefficients data to be overwritten
    open_pdf=False,  # This option only works for me right now. Please do not use it or it will spam my computer haha!
)

print(reg_table.tables[0].regressions[-1].results)  # print results from coefficients
