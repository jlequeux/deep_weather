import csv
import os


def write_results_csv(filename, fieldname, results):

    if not os.path.exists(filename):
        csvfile = open(filename, 'wb')
        writer = csv.DictWriter(csvfile, fieldnames=fieldname)
        writer.writeheader()
    else:
        csvfile = open(filename, 'ab')
        writer = csv.DictWriter(csvfile, fieldnames=fieldname)

    writer.writerow(results)
    print('DONE - Results succesfully written in %s' % filename)
    csvfile.close()
    return
