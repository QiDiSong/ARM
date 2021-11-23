移除NMS层

```
def modify_NMS_layer_IR(self):
    # step1: get the output_tensor name by spt
    self.graph_path = Pathlib(self.wkdir) / (self.model_name + '_quant.txt')
    g = Graph.parse(str(self.graph_path))
    layer_top = []
    remove_list = ['NMS', 'DecodeBox', 'Region', 'Region_fuse', 'classifier/MatMul', 'Mean_reshape']
    for node in g.nodes:
        node_layer_bottom = [input.name for input in node.inputs]
        node_layer_top = [output.name for output in node.outputs]
        node_layer_name = node.name
        if node_layer_name not in remove_list:
            layer_top.extend(node_layer_top)
            for nlb in node_layer_bottom:
                if nlb in layer_top:
                    layer_top.remove(nlb)
    output_tensor_replace = 'output_tensors=[' + ','.join(layer_top) + ']\n'


    # quant_name = self.wkdir / (self.model_name + '_quant.txt')             # old file name
    # with open(quant_name, 'r') as f:
    #     line = f.readline()
    #     while line:
    #         if 'layer_number' in line:
    #             lay_num = int(line.split('=')[1])
    #             test_id = str(lay_num - 2)
    #         line = f.readline()
    # cmd = f'aipuspt --network {self.model_name} --mode {mode_grad} --code {mode_code} -i {input_bin} --test_id {test_id} --target {self.target} -o {self.wkdir}'
    # subprocess.run(cmd, shell=True, timeout = 1200, cwd = wkdir)
    
    # # step2: open the def file
    # def_name = '0-' + test_id + '.def'
    # open_file_path = self.wkdir / self.model_name / def_name
    # output_tensor_replace = ''
    # with open(open_file_path, 'r') as f:
    #     line = f.readline()
    #     while line:
    #         if 'output_tensors' in line:
    #             output_tensor_replace = line
    #         line = f.readline()
    # f.close()

    # step3: replace the output tensor in *_quant.txt
    # write the replace str to new file, delete the old file and rename the new file
    quant_name = self.wkdir / (self.model_name + '_quant.txt')      # old file name
    backup = self.wkdir / (self.model_name + '.bak')                # new file name
    layer_end_id = 'None'
    with open(quant_name, 'r') as f1, open(backup, 'w') as f2:
        line = f1.readline()
        while line:
            if 'layer_number' in line:
                lay_num = int(line.split('=')[1])
                layer_end_id = 'layer_id=' + str(lay_num - 2)
                line = 'layer_number=' + str(lay_num - 2) + '\n'
            if layer_end_id in line:
                break
            if 'output_tensors' in line:
                line = line.replace(str(line), output_tensor_replace)
            f2.write(line)
            line = f1.readline()
    f1.close()
    f2.close()
    os.remove(quant_name)
    os.rename(backup, quant_name)
    print()
```

