import csv


def exportResults(results, labelsCsvFile, outputCsvFile):
    """
    Export classes probability in alphabetical order
    """
    labels = set()
    with open(labelsCsvFile, 'r') as labelsFile:
        reader = csv.reader(labelsFile, delimiter=',')
        next(reader)
        for row in reader:
            labels.add(row[1])

    with open(outputCsvFile, 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['id']
        header.extend(sorted(list(labels)))
        writer.writerow(header)
        for result in results:
            # Result is a tuple (id, ndarray with probabilities)
            values = result[1].tolist()
            row = [result[0]]
            row.extend(values)
            writer.writerow(row)
