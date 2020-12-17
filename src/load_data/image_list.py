def get_image_list(file:str) -> tuple:
    '''
    @params
    file: 文本文件的绝对路径
    @return
    （ (文件名1, 标签1), (文件名2, 标签2), ... ）
    '''
    image_list = []
    for line in open(file):
        line_list =  line.split( )
        if len(line_list) == 1:
            image_list.append(line_list[0])
        else:
            img, label = line_list
            image_list.append(tuple([img, int(label)]))
    return tuple(image_list)