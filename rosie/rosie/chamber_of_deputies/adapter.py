import os
from datetime import date

import numpy as np
import pandas as pd

from serenata_toolbox.chamber_of_deputies.reimbursements import Reimbursements
from serenata_toolbox.datasets import fetch


COLUMNS = {
    'category': 'subquota_description',
    'net_value': 'total_net_value',
    'recipient_id': 'cnpj_cpf',
    'recipient': 'supplier',
}
DTYPE = {
    'applicant_id': np.str,
    'cnpj_cpf': np.str,
    'congressperson_id': np.str,
    'subquota_number': np.str
}


class Adapter:
    COMPANIES_DATASET = '2016-09-03-companies.xz'

    def __init__(self, path):
        self.path = path  # directory in which datasets are saved
        self._paths = None  # reimbursement datasets by year

    @property
    def dataset(self):
        self.update_datasets()
        self.get_reimbursements()
        companies = self.get_companies()
        self._dataset = self._dataset.merge(companies,
                                            how='left',
                                            left_on='cnpj_cpf',
                                            right_on='cnpj')
        self.prepare_dataset()
        return self._dataset

    def prepare_dataset(self):
        self.rename_columns()
        self.rename_categories()

    def rename_columns(self):
        columns = {v: k for k, v in COLUMNS.items()}
        self._dataset.rename(columns=columns, inplace=True)

    def rename_categories(self):
        # There's no documented type for `3`, `4` and `5`, thus we assume it's
        # an input error until we hear back from chamber of deputies
        converters = {
                3: None,
                4: None,
                5: None,
            }
        self._dataset['document_type'].replace(converters, inplace=True)
        self._dataset['document_type'].replace(converters, inplace=True)
        self._dataset['document_type'] = self._dataset['document_type'].astype(
            'category')
        types = ['bill_of_sale', 'simple_receipt', 'expense_made_abroad']
        self._dataset['document_type'].cat.rename_categories(
            types, inplace=True)
        # Classifiers expect a more broad category name for meals
        self._dataset['category'] = self._dataset['category'].replace(
            {'Congressperson meal': 'Meal'})
        self._dataset['is_party_expense'] = \
            self._dataset['congressperson_id'].isnull()

    def update_datasets(self, years=None):
        os.makedirs(self.path, exist_ok=True)
        if not years:
            next_year = date.today().year + 1
            years = range(2009, next_year)

        fetch(self.COMPANIES_DATASET, self.path)
        self._paths = tuple(Reimbursements(str(y), self.path) for y in years)

    def get_reimbursements(self, years=None):
        self._dataset = pd.DataFrame()
        for path in self._paths:
            df = pd.read_csv(path, dtype=DTYPE, low_memory=False)
            self._dataset = pd.concat(self._dataset, df)

        self._dataset['issue_date'] = pd.to_datetime(
            self._dataset['issue_date'], errors='coerce')
        return self._dataset

    def get_companies(self):
        path = os.path.join(self.path, self.COMPANIES_DATASET)
        dataset = pd.read_csv(path, dtype={'cnpj': np.str}, low_memory=False)
        dataset['cnpj'] = dataset['cnpj'].str.replace(r'\D', '')
        dataset['situation_date'] = pd.to_datetime(
            dataset['situation_date'], errors='coerce')
        return dataset
