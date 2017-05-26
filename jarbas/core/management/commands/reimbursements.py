import csv
import lzma

from django.utils.timezone import now

from jarbas.core.management.commands import LoadCommand
from jarbas.core.models import Reimbursement


class Command(LoadCommand):
    help = 'Load Serenata de Amor reimbursements dataset'

    def handle(self, *args, **options):
        self.started_at = now()
        self.path = options['dataset']
        self.count = Reimbursement.objects.count()
        print('Starting with {:,} reimbursements'.format(self.count))

        if options.get('drop', False):
            self.drop_all(Reimbursement)
            self.count = 0

        self.create_or_update(self.reimbursements)
        self.print_count(Reimbursement, count=self.count, permanent=True)
        self.mark_not_updated_reimbursements()

    @property
    def reimbursements(self):
        """Returns a Generator with a dict object for each row."""
        with lzma.open(self.path, mode='rt') as file_handler:
            for row in csv.DictReader(file_handler):
                yield self.serialize(row)

    def serialize(self, reimbursement):
        """Read the dict generated by DictReader and fix content types"""

        missing = ('probability', 'suspicions')
        for key in missing:
            reimbursement[key] = None

        rename = (
            ('subquota_number', 'subquota_id'),
            ('reimbursement_value_total', 'total_reimbursement_value')
        )
        for old, new in rename:
            reimbursement[new] = reimbursement[old]
            del reimbursement[old]

        integers = (
            'applicant_id',
            'batch_number',
            'congressperson_document',
            'congressperson_id',
            'document_id',
            'document_type',
            'installment',
            'month',
            'subquota_group_id',
            'subquota_id',
            'term',
            'term_id',
            'year'
        )
        for key in integers:
            reimbursement[key] = self.to_number(reimbursement[key], int)

        floats = (
            'document_value',
            'remark_value',
            'total_net_value',
            'total_reimbursement_value'
        )
        for key in floats:
            reimbursement[key] = self.to_number(reimbursement[key])

        reimbursement['issue_date'] = self.to_date(reimbursement['issue_date'])

        return reimbursement

    def create_or_update(self, reimbursements_as_dicts):
        for count, reimbursement in enumerate(reimbursements_as_dicts):
            document_id = reimbursement.get('document_id')
            if document_id:
                Reimbursement.objects.update_or_create(
                    document_id=document_id,
                    defaults=reimbursement
                )
            self.print_count(Reimbursement, count=count + 1)

    def mark_not_updated_reimbursements(self):
        qs = Reimbursement.objects.filter(last_update__lt=self.started_at)
        qs.update(available_in_latest_dataset=False)
