"""The local run preview must distinguish project, decisions and stale audits."""
import unittest
from unittest.mock import patch
import dataset_curator_ui as ui


class AuditOverview(unittest.TestCase):
    def preview(self, state=None, profile=None, current=True, mode='profile_then_caption', stage=None):
        workspace={'preflight':{'completed':True,'survivor_count':3},'frame':{'enabled':True}}
        with patch.object(ui,'load_project_workspace',return_value=workspace), \
             patch.object(ui,'_workspace_preflight_is_current',return_value=current), \
             patch.object(ui,'_load_json_dict',side_effect=[stage or {'all_rows':[]}, profile or {}]), \
             patch.object(ui,'UI_LANG','de'):
            return ui.audit_run_overview_ui('example', '/dataset', 20, '<model>', mode, state or {})

    def test_current_project_counts_preserve_manual_and_excluded_decisions(self):
        base={'confidence_level':'high','crop_type':'uniform_canvas','candidate_bbox':[10,10,90,90],'layout_flags':[]}
        records=[dict(base,user_decision='auto'),dict(base,confidence_level='medium',user_decision='auto'),dict(base,user_decision='exclude')]
        state={'output_root':ui.output_root_for('/dataset','example'),'records':records,
               'frame_mode':'auto_high_review_medium','auto_accept_types':['uniform_canvas']}
        result=self.preview(state)
        self.assertIn('1 automatisch · 1 offen · 1 ausgeschlossen',result)
        self.assertIn('Captions und Export separat starten',result)
        self.assertIn('&lt;model&gt;',result)

    def test_foreign_or_stale_state_does_not_claim_current_decisions(self):
        state={'output_root':'/other','records':[{'user_decision':'exclude'}]}
        self.assertNotIn('1 ausgeschlossen',self.preview(state))
        result=self.preview(state,current=False)
        self.assertIn('Vorprüfung ausführen oder aktualisieren',result)
        self.assertNotIn('Vorprüfung bereit',result)

    def test_changed_frame_decisions_are_not_offered_as_ready_to_resume(self):
        result=self.preview(profile={'audit_stale_images':[{'source_hash':'changed'}]},mode='single_pass')
        self.assertIn('Audit-Stand muss aktualisiert',result)
        self.assertNotIn('Audit-Stand vorhanden',result)
        self.assertIn('Audit → Profil → Captions → Export',result)
        self.assertIn('Audit-Stand muss aktualisiert', self.preview(stage={'audit_stale_images':[{'source_hash':'changed'}]}))


if __name__=='__main__':
    unittest.main()
