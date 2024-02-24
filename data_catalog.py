catalog = [
    {
        "query": "What are GS1 Standards?",
        "data_product": "GS1Standards",
        "domain": "Supply Chain",
        "ownership": "Supply Chain Management Team",
        "sourceOfTruth": "GS1",
        "description": "GS1 Standards provide a common language for identifying, capturing, and sharing supply chain data, ensuring global consistency across products, locations, and logistic units.",
        "data": {
            'Standards': ['GTIN', 'SSCC', 'GLN', 'GS1 XML'],
            'Description': [
                'Global Trade Item Number: Unique identification of trade items.',
                'Serial Shipping Container Code: Identifying logistics units.',
                'Global Location Number: Identifying locations.',
                'GS1 XML: XML-based standards for business messaging.'
            ]
        }
    },
    {
        "query": "Can you describe EDI standards?",
        "data_product": "EDIStandards",
        "domain": "Supply Chain",
        "ownership": "IT Department",
        "sourceOfTruth": "ANSI X12 and EDIFACT Documentation",
        "description": "EDI standards facilitate electronic business document exchange, including purchase orders and invoices, enhancing efficiency and reducing errors in supply chain communications.",
        "data": {
            'Standards': ['ANSI X12', 'EDIFACT'],
            'Description': [
                'ANSI X12: EDI standard used in North America.',
                'EDIFACT: International EDI standard for cross-border transactions.'
            ]
        }
    },
    {
        "query": "What is VICS?",
        "data_product": "VICSStandards",
        "domain": "Supply Chain",
        "ownership": "Supply Chain Standards Committee",
        "sourceOfTruth": "VICS Documentation",
        "description": "VICS provides guidelines for EDI implementation and catalog item synchronization, improving data accuracy and synchronization across the retail supply chain.",
        "data": {
            'Standards': ['VICS EDI guidelines', 'VICS Catalogue Item Synchronization'],
            'Description': [
                'EDI guidelines for retail supply chain data synchronization.',
                'Standards for product data synchronization between retailers and suppliers.'
            ]
        }
    },
    {
        "query": "Explain NRTS.",
        "data_product": "NRTSStandards",
        "domain": "Supply Chain",
        "ownership": "National Retail Federation",
        "sourceOfTruth": "NRTS Documentation",
        "description": "NRTS offers XML-based standards for exchanging product data and inventory levels, facilitating efficient communication between retailers and their partners.",
        "data": {
            'Standards': ['Nationwide Retail Transfer Standard'],
            'Description': [
                'XML-based standards for product data and inventory level exchange.'
            ]
        }
    },
    {
        "query": "How are ASN and RA managed?",
        "data_product": "ASNandRA",
        "domain": "Logistics",
        "ownership": "Logistics Department",
        "sourceOfTruth": "Supply Chain Operations Reference",
        "description": "Advance Ship Notices and Receiving Advice documents streamline receiving and inventory management by providing detailed information on shipments and received goods.",
        "data": {
            'Documents': ['Advance Ship Notice (ASN)', 'Receiving Advice (RA)'],
            'Description': [
                'ASN: Provides detailed info about an upcoming shipment.',
                'RA: Acknowledges the receipt of goods and reconciles discrepancies.'
            ]
        }
    },
    {
        "query": "Describe POS data exchange standards.",
        "data_product": "POSDataExchange",
        "domain": "Retail",
        "ownership": "Retail Operations Team",
        "sourceOfTruth": "Retail Management Systems",
        "description": "Standards for sharing point-of-sale data to enable real-time sales tracking and inventory management, fostering collaborative planning and replenishment.",
        "data": {
            'Standards': ['EDI', 'GS1 XML'],
            'Benefits': [
                'Enables real-time sales tracking.',
                'Facilitates inventory management.'
            ]
        }
    },
    {
        "query": "What is the customer schema?",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Relations Team",
        "sourceOfTruth": "CRM System",
        "description": "Detailed customer profiles including purchase history and engagement metrics, vital for personalized marketing and customer support.",
        "data": {
            'Table': ['CustomerInfo', 'EngagementScores'],
            'Columns': [['CustomerID', 'Name', 'Email'], ['CustomerID', 'Score']],
            'Privacy': [['PII', 'PII', 'PII'], ['-', 'Confidential']],
            'Tags': [['Core', 'Core', 'Core'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "Can you describe the product schema?",
        "data_product": "ProductCatalog",
        "domain": "Inventory",
        "ownership": "Product Management Team",
        "sourceOfTruth": "Product Database",
        "description": "Comprehensive product listings, including stock levels and categories, enabling inventory management and sales strategy planning.",
        "data": {
            'Table': ['ProductInfo', 'StockLevels'],
            'Columns': [['ProductID', 'Name', 'Category'], ['ProductID', 'StockAvailable']],
            'Privacy': [['-', '-', '-'], ['-', '-']],
            'Tags': [['Core', 'Core', 'Attribute'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "Show the sales lineage.",
        "data_product": "SalesData",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Sales Database",
        "description": "Tracks the journey of sales data from order capture to final analysis, critical for understanding sales trends and customer preferences.",
        "data": {
            'source': ['OrderInfo', 'CustomerInfo'],
            'target': ['SalesAnalysis', 'CustomerSegmentation'],
            'Columns': ['OrderID, ProductID, CustomerID', 'CustomerID, Segment'],
            'Table and Columns': [
                'OrderInfo -> SalesAnalysis: OrderID, ProductID, CustomerID',
                'CustomerInfo -> CustomerSegmentation: CustomerID, Segment'
            ]
        }
    },
    {
        "query": "What is the inventory readiness?",
        "data_product": "InventoryManagement",
        "domain": "Inventory",
        "ownership": "Inventory Control Team",
        "sourceOfTruth": "Warehouse Management System",
        "description": "Assesses inventory accuracy and availability, ensuring stock levels meet demand without overstocking, essential for efficient supply chain management.",
        "data": {
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [98, 99, 97, 95]
        }
    },
    {
        "query": "What are the customer usage patterns?",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Insights Team",
        "sourceOfTruth": "Usage Tracking System",
        "description": "Analyzes customer interaction and purchasing patterns over time to refine product offerings and marketing strategies.",
        "data": {
            'Month': ['January', 'February', 'March', 'April'],
            'Usage': [220, 230, 210, 250]
        }
    },
    {
        "query": "Explain the POS data exchange standards.",
        "data_product": "POSDataExchange",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Retail POS Systems",
        "description": "Facilitates the seamless exchange of sales data between retail points of sale and corporate systems, enhancing real-time sales tracking and inventory management.",
        "data": {
            'Standards': ['EDI', 'GS1 XML'],
            'Documents': ['Invoices', 'Purchase Orders'],
            'Frequency': ['Daily', 'Weekly'],
            'Partners': ['Retailers', 'Suppliers']
        }
    },
    {
        "query": "How is shipment tracking managed in the logistics domain?",
        "data_product": "ShipmentTracking",
        "domain": "Logistics",
        "ownership": "Logistics Team",
        "sourceOfTruth": "Logistics Systems",
        "description": "Monitors the real-time status of shipments across the supply chain, ensuring timely delivery and minimizing disruptions.",
        "data": {
            'Standards': ['GS1 Standards', 'EDI'],
            'Documents': ['Advance Ship Notices (ASN)', 'Receiving Advice (RA)'],
            'Frequency': ['Real-time', 'Periodic'],
            'Partners': ['Carriers', 'Warehouses']
        },

    }
]


# Define personas
personas = {
    "Data Steward": ["What compliance standards are followed for supply chain data exchanges?",
                     "Detail the audit processes for data governance compliance.",
                     "Explain the role and responsibilities of the data governance committee.",
                     "Workflows"],
    "Data Guardian": ["Outline the data privacy policies for customer information.",
                      "What are the security protocols for product data?",
                      "How is sensitive data identified and protected in customer profiles.",
                      "What are the procedures for data breach response and notification?"],
    "Engineer": ["What is the customer schema?",
                 "Can you describe the product schema?",
                 "Show the sales lineage.",
                 "What is the inventory readiness?",
                 "What are the customer usage patterns?",
                 "Explain the POS data exchange standards.",
                 "How is shipment tracking managed in the logistics domain?",
                 "What are GS1 Standards?",
                 "Can you describe EDI standards?",
                 "What is VICS?",
                 "Explain NRTS.",
                 "How are ASN and RA managed?",
                 "Describe POS data exchange standards.",
                 "How is data quality in inventory management ensured?",
                 "Explain the consent management process for customer data usage.",
                 "Describe the governance framework for sales data lineage.",
                 "Describe the process for third-party data sharing and agreements.",
                 "How is anonymization applied to sales and customer data for analysis?"]
}


# Define catalog structure text
catalog_structure_text = """
The data catalog consists of various domains, each with specific data products owned by designated teams and linked to their respective sources of truth. 
- The **Customer Domain** includes 'Customer360' and 'CustomerUsagePattern', owned by the Customer Relations Team and Customer Insights Team, respectively, with CRM and Usage Tracking Systems as sources of truth.
- The **Inventory Domain** covers 'ProductCatalog' and 'InventoryManagement', managed by the Product Management Team and Inventory Control Team, with the Product Database and Warehouse Management System as sources of truth.
- Under the **Sales Domain**, 'SalesData' and 'POSDataExchange' are analyzed by the Sales Analysis Team and rely on the Sales Database and Retail POS Systems.
- The **Logistics Domain** offers 'ShipmentTracking', owned by the Logistics Team with the Logistics Systems as the source of truth.
"""

# Pretrained model selection
model_options = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base"
}