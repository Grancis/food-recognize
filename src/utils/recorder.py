import config

def train_recorder(record_list:tuple, header=False):
    '''
    @params
    record_list: ( (epoch1, step1, loss1), (epoch1, step2, loss2) ...)
    '''
    with open(config.RECORD_TRAIN_PATH, 'a') as record_f:
        record_f = open(config.RECORD_TRAIN_PATH, 'a')
        if header:
            record_f.write('Epoch\tstep\tloss\n')
        for record in record_list:
            record_f.write(str(record[0])+'\t'+str(record[1])+'\t'+str(record[2])+'\n')

def validate_recorder(record_list:tuple, header=False):
    '''
    @params
    record_list: ( (epoch1, step1, loss1), (epoch1, step2, loss2) ...)
    '''
    with open(config.RECORD_VALIDATE_PATH, 'a') as record_f:
        record_f = open(config.RECORD_VALIDATE_PATH, 'a')
        if header:
            record_f.write('Epoch\tstep\tloss\n')
        for record in record_list:
            record_f.write(str(record[0])+'\t'+str(record[1]+'\t'+record[2]+'\n'))