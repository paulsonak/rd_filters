#!/usr/bin/env python3
"""
Adapted from Pat Walters to be used with jupyter
Added several functions to visualize molecules and filters
AKP 10/21/2022
"""
#################################################
import sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors, TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem
from rdkit.Chem.rdFMCS import FindMCS
import multiprocessing as mp
from multiprocessing import Pool
import time
import pandas as pd
import os
import json
from docopt import docopt
import pkg_resources
import logging
logging.basicConfig(format='%(asctime)-15s %(message)s')
logger = logging.getLogger()


cmd_str = """Usage:
rd_filters filter --in INPUT_FILE --prefix PREFIX [--rules RULES_FILE_NAME] [--alerts ALERT_FILE_NAME][--np NUM_CORES]
rd_filters template --out TEMPLATE_FILE [--rules RULES_FILE_NAME]

Options:
--in INPUT_FILE input file name
--prefix PREFIX prefix for output file names
--rules RULES_FILE_NAME name of the rules JSON file
--alerts ALERTS_FILE_NAME name of the structural alerts file
--np NUM_CORES the number of cpu cores to use (default is all)
--out TEMPLATE_FILE parameter template file name
"""

def get_rules_files(rd_filter_data_path):
    rules={}
    for file in os.listdir(rd_filter_data_path):
        if '.json' in file:
            rules[file.replace('.json','')]=file
    return rules

def read_rules(rules_file_name):
    """
    Read rules from a JSON file
    :param rules_file_name: JSON file name
    :return: dictionary corresponding to the contents of the JSON file
    """
    with open(rules_file_name) as json_file:
        try:
            rules_dict = json.load(json_file)
            return rules_dict
        except json.JSONDecodeError:
            print(f"Error parsing JSON file {rules_file_name}")
            sys.exit(1)


def write_rules(rule_dict, file_name):
    """
    Write configuration to a JSON file
    :param rule_dict: dictionary with rules
    :param file_name: JSON file name
    :return: None
    """
    ofs = open(file_name, "w")
    ofs.write(json.dumps(rule_dict, indent=4, sort_keys=True))
    print(f"Wrote rules to {file_name}")
    ofs.close()


def default_rule_template(alert_list, file_name):
    """
    Build a default rules template
    :param alert_list: list of alert set names
    :param file_name: output file name
    :return: None
    """
    default_rule_dict = {
        "MW": [0, 500],
        "LogP": [-5, 5],
        "HBD": [0, 5],
        "HBA": [0, 10],
        "TPSA": [0, 200],
        "Rot": [0, 10]
    }
    for rule_name in alert_list:
        if rule_name == "Inpharmatica":
            default_rule_dict["Rule_" + rule_name] = True
        else:
            default_rule_dict["Rule_" + rule_name] = False
    write_rules(default_rule_dict, file_name)


def get_config_file(file_name, environment_variable):
    """
    Read a configuration file, first look for the file, if you can't find
    it there, look in the directory pointed to by environment_variable
    :param file_name: the configuration file
    :param environment_variable: the environment variable
    :return: the file name or file_path if it exists otherwise exit
    """
    if os.path.exists(file_name):
        return file_name
    else:
        config_dir = os.environ.get(environment_variable)
        if config_dir:
            config_file_path = os.path.join(os.path.sep, config_dir, file_name)
            if os.path.exists(config_file_path):
                return config_file_path

    error_list = [f"Could not file {file_name}"]
    if config_dir:
        err_str = f"Could not find {config_file_path} based on the {environment_variable}" + \
                  "environment variable"
        error_list.append(err_str)
    error_list.append(f"Please check {file_name} exists")
    error_list.append(f"Or in the directory pointed to by the {environment_variable} environment variable")
    print("\n".join(error_list))
    sys.exit(1)


class RDFilters:
    def __init__(self, rules_file_name):
        good_name = get_config_file(rules_file_name, "FILTER_RULES_DIR")
        self.rule_df = pd.read_csv(good_name)
        # make sure there wasn't a blank line introduced
        self.rule_df = self.rule_df.dropna()
        self.rule_list = []

    def build_rule_list(self, alert_name_list):
        """
        Read the alerts csv file and select the rule sets defined in alert_name_list
        :param alert_name_list: list of alert sets to use
        :return:
        """
        self.rule_df = self.rule_df[self.rule_df.rule_set_name.isin(alert_name_list)]
        tmp_rule_list = self.rule_df[["rule_id", "smarts", "max", "description"]].values.tolist()
        for rule_id, smarts, max_val, desc in tmp_rule_list:
            smarts_mol = Chem.MolFromSmarts(smarts)
            if smarts_mol:
                self.rule_list.append([smarts_mol, max_val, desc])
            else:
                print(f"Error parsing SMARTS for rule {rule_id}", file=sys.stderr)

    def get_alert_sets(self):
        """
        :return: a list of unique rule set names
        """
        return self.rule_df.rule_set_name.unique()

    def evaluate(self, lst_in):
        """
        Evaluate structure alerts on a list of SMILES
        :param lst_in: input list of [SMILES, Name]
        :return: list of alerts matched or "OK"
        """
        smiles, name = lst_in
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles, name, 'INVALID', -999, -999, -999, -999, -999, -999]
        desc_list = [MolWt(mol), MolLogP(mol), NumHDonors(mol), NumHAcceptors(mol), TPSA(mol),
                     CalcNumRotatableBonds(mol)]
        for row in self.rule_list:
            patt, max_val, desc = row
            if len(mol.GetSubstructMatches(patt)) > max_val:
                return [smiles, name] + [desc + " > %d" % (max_val)] + desc_list
        return [smiles, name] + ["OK"] + desc_list

##################################################################
##### AKP modified main() ########################################

def filter_mols(cmd_input, verbose=True):
    """ A function to replicate the command line functionality in a jupyter notebook.
    Args:
        cmd_input (dict): input dict of options {
            (template|filter):boolean, 
            out: a file location for creating a template,
            in: an input .smi file, 
            data_path: a path to the directory of alerts and template files, 
            rules: label for which template file to use, 
            prefix: an indicator of where to save filtering info and what to call it, 
            num_cores: number of cores to use with default all cores, 
            alerts: a file name indicating which alerts to use instead of a template
            }
        verbose (bool): whether to print comments
    Returns
        None
    Effects:
        Saves template to out or filter info to prefix
    """
    alert_file_name = cmd_input.get("alerts") or pkg_resources.resource_filename('rd_filters', "data/alert_collection.csv")
    rf = RDFilters(alert_file_name)

    if cmd_input.get("template"):
        template_output_file = cmd_input.get("out")
        default_rule_template(rf.get_alert_sets(), template_output_file)

    elif cmd_input.get("filter"):
        input_file_name = cmd_input.get("in")
        data_path=cmd_input.get("data_path") or pkg_resources.resource_filename('rd_filters', "data")
        rules_files=get_rules_files(data_path)
        rules_file_name = os.path.join(data_path, rules_files[cmd_input.get("rules")]) or pkg_resources.resource_filename('rd_filters', "data/rules_all.json")
        rules_file_path = get_config_file(rules_file_name, "FILTER_RULES_DATA")
        prefix_name = cmd_input.get("prefix")
        num_cores = cmd_input.get("num_cores") or mp.cpu_count()
        num_cores = int(num_cores)

        if verbose:
            print("using %d cores" % num_cores, file=sys.stderr)
        start_time = time.time()
        p = Pool(num_cores)
        input_data = [x.split() for x in open(input_file_name)]
        input_data = [x for x in input_data if len(x) == 2]
        rule_dict = read_rules(rules_file_path)

        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        if verbose:
            print(f"Using alerts from {rule_str}", file=sys.stderr)
        rf.build_rule_list(rule_list)
        res = list(p.map(rf.evaluate, input_data))
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*rule_dict["MW"]) &
            df.LogP.between(*rule_dict["LogP"]) &
            df.HBD.between(*rule_dict["HBD"]) &
            df.HBA.between(*rule_dict["HBA"]) &
            df.TPSA.between(*rule_dict["TPSA"]) &
            df.Rot.between(*rule_dict["Rot"])
            ]
        output_smiles_file = prefix_name + ".smi"
        output_csv_file = prefix_name + ".csv"
        df_ok[["SMILES", "NAME"]].to_csv(f"{output_smiles_file}", sep=" ", index=False, header=False)
        if verbose:
            print(f"Wrote SMILES for molecules passing filters to {output_smiles_file}", file=sys.stderr)
        df.to_csv(f"{prefix_name}.csv", index=False)
        if verbose:
            print(f"Wrote detailed data to {output_csv_file}", file=sys.stderr)

        num_input_rows = df.shape[0]
        num_output_rows = df_ok.shape[0]
        fraction_passed = "%.1f" % (num_output_rows / num_input_rows * 100.0)
        if verbose:
            print(f"{num_output_rows} of {num_input_rows} passed filters {fraction_passed}%", file=sys.stderr)
        elapsed_time = "%.2f" % (time.time() - start_time)
        if verbose:
            print(f"Elapsed time {elapsed_time} seconds", file=sys.stderr)
        
        
##################################################################
##################################################################
##### AKP functions ##############################################

# This function enumerates and scores tautomers for the molecule passed as parameter, puts them in a `pandas.DataFrame` and sorts it by decreasing score.
# from https://gist.github.com/ptosco/20b06985cd8830d5e549165f6b9fc969
def get_tautomer_dataframe(mol):
    te = rdMolStandardize.TautomerEnumerator()
    te.SetMaxTransforms(2000)
    res = te.Enumerate(mol)
    df = pd.DataFrame({"TautSmiles": [Chem.MolToSmiles(t) for t in res], "TautScore": [te.ScoreTautomer(t) for t in res]})
    Chem.PandasTools.AddMoleculeColumnToFrame(df, "TautSmiles")
    df.sort_values(by=["TautScore", "TautSmiles"], ascending=[False, True], inplace=True)
    return df

# function to align two molecule images based on RDKit cookbook
def align_mol(template, mol2):
    AllChem.Compute2DCoords(template)
    mcs = FindMCS([template,mol2], 
                  completeRingsOnly=True, 
                  ringMatchesRingOnly=True, 
                  timeout=10)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    template_match = template.GetSubstructMatch(patt)
    query_match = mol2.GetSubstructMatch(patt)
    AllChem.Compute2DCoords(mol2)
    rms = AllChem.AlignMol(mol2, template, atomMap=list(zip(query_match,template_match)))
    
# draw original structure, best tautomer, and a corrected/"best" structure if exists
def draw_structures(smiles_dict, corr_smiles_dict, save=False, outdir='./'):
    for root in smiles_dict:
        smi = smiles_dict[root]
        mol = Chem.MolFromSmiles(smi)
        canmol=rdMolStandardize.CanonicalTautomer(mol)
        if root in corr_smiles_dict.keys():
            corsmi=corr_smiles_dict[root]
            cormol=Chem.MolFromSmiles(corsmi)
            align_mol(canmol, cormol)
        else:
            cormol=Chem.MolFromSmiles('[Na+]')
        mols=[mol,canmol,cormol]
        legends=[root,'Canonical tautomer','Jeff corrected']
        svg=AllChem.Draw.MolsToGridImage(mols, legends=legends, molsPerRow=3, subImgSize=(300,300),highlightAtomLists=None, highlightBondLists=None, useSVG=True)
        display(svg)
        if save:
            filename=os.path.join(outdir, f'{root}.svg')
            with open(filename, 'w') as f:
                    f.write(svg.data)

def draw_filters(filterdict):
    for label, smarts in filterdict.items():
        mol=Chem.MolFromSmarts(smarts)
        print(label)
        display(mol)

def get_fail_smarts(filter_string, filters):
    filter_string=filter_string.split(' > ')
    if len(filter_string)>2:
        logger.warning("This molecule has more than one alert, only returning the first one.")
    filter_string=filter_string[0]
    smarts_list =  filters[filters.description==filter_string].smarts.tolist()
    return smarts_list

# https://stackoverflow.com/questions/69735586/how-to-highlight-the-substructure-of-a-molecule-with-thick-red-lines-in-rdkit-as
def increase_resolution(mol, substructure, size=(400, 400)):
    # mol = deepcopy(mol)
    # substructure = deepcopy(substructure)
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    # highlightAtoms expects only one tuple, not tuple of tuples. So it needs to be merged into a single tuple
    matches = sum(mol.GetSubstructMatches(substructure), ())
    drawer.DrawMolecule(mol, highlightAtoms=matches)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('xmlns:svg','xmlns')
    return svg.replace('svg:','')

def draw_filter_on_mol(smiles, filter_string, filters_df):
    m = Chem.MolFromSmiles(smiles)
    smarts=get_fail_smarts(filter_string, filters_df)
    if len(smarts)==0:
        smarts=['']
    substructure = Chem.MolFromSmarts(smarts[0])
    # m=increase_resolution(m, substructure)
    matches=m.GetSubstructMatches(substructure)
    return m

def get_filter_df(data_path, rules=['SureChEMBL']):
    if os.path.isfile(data_path):
        filter_df=pd.read_csv(data_path)
    else:
        filter_df=pd.read_csv(pkg_resources.resource_filename('rd_filters', "data/alert_collection.csv"))
        if rules[0]!= 'all':
            filter_df=filter_df[filter_df.rule_set_name.isin(rules)]
    return filter_df

def add_filtered_mol_col_to_df(df, smiles_col, filter_col, data_path='', rules=['SureChEMBL'], mol_col='FilteredMol'):
    filter_df=get_filter_df(data_path, rules)
    drawn_mols=[]
    match_list=[]
    for smiles, filter_string in zip(df[smiles_col], df[filter_col]):
        mol = draw_filter_on_mol(smiles, filter_string, filter_df)
        drawn_mols.append(mol)
    df[mol_col]=drawn_mols
    
def add_filter_col_to_df(df, df_filter_col, data_path='', rules=['SureChEMBL'], mol_col='FilterMol'):
    """ Function to add an image of the filter to your df.
    Args:
        df (Pandas.DataFrame): df to add
        df_filter_col (str): name of the column containing the filter name in df
        data_path (path): path to list of alerts, either this or rules can be specified. Must be a csv with 'description' matching the filter name and 'smarts' with smarts.
        rules (list): list of rules sets included in rd_filters package, or ['all']
        mol_col: name of column containing filter mol image
    Returns:
        None
    Effects:
        Adds mol_col to df in place.
    """
    filter_df=get_filter_df(data_path, rules)               
    filtmols=[]
    smartslist=[]
    for filter_string in df[df_filter_col]:
        smarts=get_fail_smarts(filter_string, filter_df)
        if len(smarts)==0:
            smarts=['']
        filtmols.append(Chem.MolFromSmarts(smarts[0]))
        smartslist.append(smarts[0])
    df['smarts']=smartslist
    df[mol_col]=filtmols
    

##################################################################
##################################################################


def main():
    cmd_input = docopt(cmd_str)
    alert_file_name = cmd_input.get("--alerts") or pkg_resources.resource_filename('rd_filters',
                                                                                   "data/alert_collection.csv")
    rf = RDFilters(alert_file_name)

    if cmd_input.get("template"):
        template_output_file = cmd_input.get("--out")
        default_rule_template(rf.get_alert_sets(), template_output_file)

    elif cmd_input.get("filter"):
        input_file_name = cmd_input.get("--in")
        rules_file_name = cmd_input.get("--rules") or pkg_resources.resource_filename('rd_filters', "data/rules.json")
        rules_file_path = get_config_file(rules_file_name, "FILTER_RULES_DATA")
        prefix_name = cmd_input.get("--prefix")
        num_cores = cmd_input.get("--np") or mp.cpu_count()
        num_cores = int(num_cores)

        print("using %d cores" % num_cores, file=sys.stderr)
        start_time = time.time()
        p = Pool(num_cores)
        input_data = [x.split() for x in open(input_file_name)]
        input_data = [x for x in input_data if len(x) == 2]
        rule_dict = read_rules(rules_file_path)

        rule_list = [x.replace("Rule_", "") for x in rule_dict.keys() if x.startswith("Rule") and rule_dict[x]]
        rule_str = " and ".join(rule_list)
        print(f"Using alerts from {rule_str}", file=sys.stderr)
        rf.build_rule_list(rule_list)
        res = list(p.map(rf.evaluate, input_data))
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*rule_dict["MW"]) &
            df.LogP.between(*rule_dict["LogP"]) &
            df.HBD.between(*rule_dict["HBD"]) &
            df.HBA.between(*rule_dict["HBA"]) &
            df.TPSA.between(*rule_dict["TPSA"]) &
            df.Rot.between(*rule_dict["Rot"])
            ]
        output_smiles_file = prefix_name + ".smi"
        output_csv_file = prefix_name + ".csv"
        df_ok[["SMILES", "NAME"]].to_csv(f"{output_smiles_file}", sep=" ", index=False, header=False)
        print(f"Wrote SMILES for molecules passing filters to {output_smiles_file}", file=sys.stderr)
        df.to_csv(f"{prefix_name}.csv", index=False)
        print(f"Wrote detailed data to {output_csv_file}", file=sys.stderr)

        num_input_rows = df.shape[0]
        num_output_rows = df_ok.shape[0]
        fraction_passed = "%.1f" % (num_output_rows / num_input_rows * 100.0)
        print(f"{num_output_rows} of {num_input_rows} passed filters {fraction_passed}%", file=sys.stderr)
        elapsed_time = "%.2f" % (time.time() - start_time)
        print(f"Elapsed time {elapsed_time} seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
