if __name__ == '__main__':
    import sys
    import numpy as np
    import array

    if len(sys.argv) < 2:
        print "Please input the file path"
        sys.exit()
    file_path = sys.argv[1]

    with open(file_path) as f:

        training_sets = []
        target_sets = []

        for line in f:
            single_line = line.split(",")
            training_sets.append( map( lambda i: float(i), single_line[:-1] ))
            target_sets.append( int(single_line[-1]))

        # convert to numpy array
        # training_sets = np.array(training_sets)
        # target_sets = np.array(target_sets)

        # write training data and target data to file
        with open('training_data.txt', 'w') as t1, open('target_data.txt', 'w') as t2:
            t1.write("%s\n" % training_sets)
            t2.write("%s\n" % target_sets)


        print training_sets

        print "-----------------"


        # print target_sets


