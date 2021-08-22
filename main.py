from nmtest.attribute_aggregator import AttributeAggregator

def main():
    attribute_aggregator: AttributeAggregator = AttributeAggregator('./data')
    attribute_aggregator.aggregate()
    attribute_aggregator.print_metrics()
        
if __name__ == '__main__':
    main()