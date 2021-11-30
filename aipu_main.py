aipu main.py
# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-


import argparse
import configparser
import os
import sys
import shlex
import re
import tempfile
import time
import torch
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import AIPUBuilder.core
import AIPUBuilder.plugin_loader as pl
from AIPUBuilder import __VERSION__
from AIPUBuilder.logger import INFO, ERROR, WARN, set_log_file


class VersioningAction(argparse.Action):
    def __init__(self, option_strings, dest,version, **kwargs):
        self.version=version
        super(VersioningAction, self).__init__(option_strings, dest,nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        os.environ["AIPUBUILDER_LOG"]="10"
        from AIPUBuilder.plugin_loader import VERSIONS
        from AIPUBuilder.Parser import __VERSION__ as parser_version
        from AIPUBuilder.Optimizer.opt_information import __OPT_VERSION__ as opt_version
        from AIPUBuilder.Optimizer import plugins
        from AIPUBuilder.core import __version__,__git__,get_plugin_version
        from AIPUBuilder.CGBuilder import aipurun_version, aipugb_version
        print(self.version)
        print("Parser   \tversion:",parser_version)
        print("Optimizer\tversion:",opt_version)
        print("Core Lib \tversion:",__version__,"\tgit commit:",__git__)
        print("aipurun  \tversion:",aipurun_version.version)
        print("aipugb   \tversion:",aipugb_version.version)
        for ptype,ps in VERSIONS.items():
            if len(ps)==0:
                continue
            print("\n%s plugins version:"%(str(ptype)[11:]))
            for name,ver in ps.items():
                print("%-24s\tversion:%s"%(name,ver))
        print(get_plugin_version())
        exit(0)


def main():
    from AIPUBuilder.Optimizer import plugins
    from AIPUBuilder.Optimizer import cfg_parser
    opt_cfg_parameters = cfg_parser.show_cfg_parameter()
    epilog = opt_cfg_parameters + "Please refer the document for more details about configuration file.\n"
    args=argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a AIPU graph from frontend framework with a config file.",
        epilog=epilog,
    )
    args.add_argument('-v', '--version', action=VersioningAction,
                      version='%-10s\tversion: '%(sys.argv[0])+__VERSION__)
    args.add_argument("config", metavar='<net.cfg>',
                      type=str,  help='graph configure file.')

    options=args.parse_args(sys.argv[1:])
    from AIPUBuilder.plugin_loader import ALL_MODULES
    from AIPUBuilder.Optimizer.tools.optimizer_main import main as opt
    from AIPUBuilder.Parser.main import main as parser
    from AIPUBuilder.CGBuilder import caipugb, caipurun
    maipurun=caipurun
    maipugb=caipugb
    config=configparser.ConfigParser()
    config.read(options.config)
    mode="run"
    if "Common" in config:
        common=config["Common"]
        if "mode" in common:
            mode=common["mode"].strip()
        if "log_file" in common:
            log_file=common["log_file"].strip()
            if log_file:
                set_log_file(log_file)
    if mode not in ["run", "build"]:
        ERROR("Config file error, mode must be run or build, now is %s" % (mode))
    temp_dir=tempfile.gettempdir()
    work_folder=os.path.join(temp_dir, "AIPUBuilder_"+str(time.time()))
    if not os.path.exists(work_folder):
        os.mkdir(work_folder)
    if "Parser" not in config:
        ERROR("Config file missing Parser section")
    parser_options=config["Parser"]
    model_name=config["Parser"]["model_name"]
    parser_cfg=configparser.ConfigParser()
    parser_cfg['Common']=config["Parser"]
    parser_out=work_folder
    if "output_dir" in config["Parser"]:
        parser_out=config["Parser"]["output_dir"]
    parser_cfg['Common']["output_dir"]=parser_out
    p_cf=os.path.join(work_folder, "parser.cfg")
    with open(p_cf, "w") as f:
        parser_cfg.write(f)
    INFO("Parsing model....")
    sys.argv=["parser", "-c", p_cf]
    try:
        parser()
        INFO("Parse model complete")
    except Exception as e:
        print(e)
        ERROR("Parse model failed! %s" % (str(e)))
    argv=["gbuilder", os.path.join(quant_ir_folder, aop['quant_ir_name'] + ".txt"),
            "-w", os.path.join(quant_ir_folder, aop['quant_ir_name'] + ".bin")]
    for k in config["GBuilder"]:
        v=config["GBuilder"][k]
        if v is not None and len(v.strip()) != 0:
            if k == "inputs":
                if mode == "run":
                    argv.append("-i")
                    argv.append(v)
            elif k == "outputs":
                argv.append("-o")
                argv.append(v)
            elif k == "local_lib":
                argv.append("-L")
                argv.append(v)
            elif k.upper() == "DEBUG":
                if v.upper() == "TRUE":
                    argv.append("-D")
            elif k == "dump":
                if v.upper() == "TRUE":
                    argv.append("--dump")
            elif k == "profile":
                if v.upper() == "TRUE":
                    argv.append("--profile")
            elif k == "prof_unit":
                if v.upper() in ["AIFF", "TPC"]:
                    argv.append("--prof_unit")
                    argv.append(v.upper())
            elif k == "disable_mmu":
                if v.upper() == "TRUE":
                    argv.append("--disable_mmu")
            elif v.upper() == "TRUE":
                argv.append("--%s" % (k))
            elif v.upper() == "FALSE":
                continue
            else:
                argv.append("--%s" % (k))
                argv.append(v)
    sys.argv[:]=argv[:]
    # print(sys.argv)
    INFO("Building ...")
    if mode == "run":
        sys.argv[0]="aipurun"
        maipurun()
    elif mode == "build":
        sys.argv[0]="aipugb"
        maipugb()
    else:
        ERROR("config error, nothing to do.")


if __name__ == "__main__":
    main()
    pass
