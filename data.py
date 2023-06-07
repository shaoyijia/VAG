import json
import os.path
import pickle

import pandas as pd
from datasets import Dataset, load_dataset

from utils import load_json, dump_json

CLINC_LABEL2ID = {"restaurant_reviews": 0, "improve_credit_score": 1, "transfer": 2, "change_language": 3,
                  "transactions": 4, "ingredients_list": 5, "ingredient_substitution": 6, "timer": 7, "definition": 8,
                  "no": 9, "direct_deposit": 10, "flip_coin": 11, "are_you_a_bot": 12, "book_flight": 13,
                  "restaurant_suggestion": 14, "repeat": 15, "user_name": 16, "report_lost_card": 17,
                  "nutrition_info": 18, "recipe": 19, "rollover_401k": 20, "travel_notification": 21, "text": 22,
                  "bill_balance": 23, "spelling": 24, "tell_joke": 25, "directions": 26, "do_you_have_pets": 27,
                  "pin_change": 28, "pto_request_status": 29, "flight_status": 30, "change_user_name": 31,
                  "credit_limit_change": 32, "cancel": 33, "pto_balance": 34, "translate": 35, "exchange_rate": 36,
                  "roll_dice": 37, "change_ai_name": 38, "sync_device": 39, "insurance_change": 40,
                  "calendar_update": 41, "expiration_date": 42, "update_playlist": 43, "pay_bill": 44,
                  "reminder_update": 45, "travel_alert": 46, "cancel_reservation": 47, "food_last": 48, "greeting": 49,
                  "calendar": 50, "gas_type": 51, "credit_score": 52, "min_payment": 53, "next_holiday": 54,
                  "calories": 55, "cook_time": 56, "what_song": 57, "international_fees": 58, "interest_rate": 59,
                  "freeze_account": 60, "maybe": 61, "lost_luggage": 62, "meaning_of_life": 63, "spending_history": 64,
                  "change_speed": 65, "redeem_rewards": 66, "reminder": 67, "todo_list": 68, "oil_change_when": 69,
                  "credit_limit": 70, "accept_reservations": 71, "shopping_list_update": 72, "meeting_schedule": 73,
                  "current_location": 74, "share_location": 75, "what_can_i_ask_you": 76, "income": 77, "insurance": 78,
                  "calculator": 79, "payday": 80, "timezone": 81, "gas": 82, "how_busy": 83, "pto_request": 84,
                  "bill_due": 85, "who_made_you": 86, "new_card": 87, "car_rental": 88, "travel_suggestion": 89,
                  "plug_type": 90, "international_visa": 91, "play_music": 92, "w2": 93, "confirm_reservation": 94,
                  "schedule_meeting": 95, "what_is_your_name": 96, "time": 97, "weather": 98, "carry_on": 99,
                  "where_are_you_from": 100, "smart_home": 101, "order": 102, "last_maintenance": 103,
                  "order_status": 104, "order_checks": 105, "how_old_are_you": 106, "damaged_card": 107,
                  "schedule_maintenance": 108, "alarm": 109, "yes": 110, "change_volume": 111,
                  "what_are_your_hobbies": 112, "vaccines": 113, "todo_list_update": 114, "routing": 115,
                  "shopping_list": 116, "taxes": 117, "next_song": 118, "book_hotel": 119, "report_fraud": 120,
                  "replacement_card_duration": 121, "change_accent": 122, "jump_start": 123, "card_declined": 124,
                  "fun_fact": 125, "tire_pressure": 126, "goodbye": 127, "uber": 128, "tire_change": 129,
                  "oil_change_how": 130, "meal_suggestion": 131, "date": 132, "make_call": 133, "balance": 134,
                  "apr": 135, "whisper_mode": 136, "traffic": 137, "measurement_conversion": 138,
                  "application_status": 139, "pto_used": 140, "account_blocked": 141, "find_phone": 142, "mpg": 143,
                  "thank_you": 144, "restaurant_reservation": 145, "distance": 146, "rewards_balance": 147,
                  "who_do_you_work_for": 148, "reset_settings": 149, "oos": 150}

BANKING_LABEL2ID = {
    'contactless_not_working': 0, 'card_payment_wrong_exchange_rate': 1, 'compromised_card': 2,
    'get_physical_card': 3, 'edit_personal_details': 4, 'cancel_transfer': 5, 'pin_blocked': 6,
    'exchange_charge': 7, 'lost_or_stolen_card': 8, 'beneficiary_not_allowed': 9, 'extra_charge_on_statement': 10,
    'verify_my_identity': 11, 'cash_withdrawal_not_recognised': 12, 'card_payment_not_recognised': 13,
    'country_support': 14, 'visa_or_mastercard': 15, 'passcode_forgotten': 16, 'card_linking': 17,
    'apple_pay_or_google_pay': 18, 'age_limit': 19, 'getting_spare_card': 20, 'receiving_money': 21,
    'verify_top_up': 22, 'card_swallowed': 23, 'pending_card_payment': 24, 'supported_cards_and_currencies': 25,
    'request_refund': 26, 'transfer_timing': 27, 'top_up_reverted': 28, 'card_arrival': 29,
    'wrong_amount_of_cash_received': 30, 'activate_my_card': 31, 'pending_transfer': 32,
    'direct_debit_payment_not_recognised': 33, 'unable_to_verify_identity': 34, 'top_up_limits': 35,
    'getting_virtual_card': 36, 'top_up_by_card_charge': 37, 'exchange_via_app': 38, 'card_not_working': 39,
    'change_pin': 40, 'pending_top_up': 41, 'verify_source_of_funds': 42, 'balance_not_updated_after_bank_transfer': 43,
    'lost_or_stolen_phone': 44, 'fiat_currency_support': 45, 'reverted_card_payment?': 46, 'atm_support': 47,
    'declined_transfer': 48, 'card_delivery_estimate': 49, 'Refund_not_showing_up': 50, 'automatic_top_up': 51,
    'failed_transfer': 52, 'topping_up_by_card': 53, 'order_physical_card': 54, 'top_up_by_cash_or_cheque': 55,
    'declined_cash_withdrawal': 56, 'transfer_into_account': 57, 'card_acceptance': 58, 'cash_withdrawal_charge': 59,
    'pending_cash_withdrawal': 60, 'disposable_card_limits': 61, 'virtual_card_not_working': 62,
    'transfer_not_received_by_recipient': 63, 'card_about_to_expire': 64, 'card_payment_fee_charged': 65,
    'get_disposable_virtual_card': 66, 'why_verify_identity': 67, 'wrong_exchange_rate_for_cash_withdrawal': 68,
    'top_up_failed': 69, 'transfer_fee_charged': 70, 'top_up_by_bank_transfer_charge': 71,
    'transaction_charged_twice': 72, 'balance_not_updated_after_cheque_or_cash_deposit': 73, 'terminate_account': 74,
    'declined_card_payment': 75, 'exchange_rate': 76
}

classes_20newsgroup = [
    "19997_comp.graphics",
    "19997_comp.os.ms-windows.misc",
    "19997_comp.sys.ibm.pc.hardware",
    "19997_comp.sys.mac.hardware",
    "19997_comp.windows.x",
    "19997_rec.autos",
    "19997_rec.motorcycles",
    "19997_rec.sport.baseball",
    "19997_rec.sport.hockey",
    "19997_sci.crypt",
    "19997_sci.electronics",
    "19997_sci.med",
    "19997_sci.space",
    "19997_misc.forsale",
    "19997_talk.politics.misc",
    "19997_talk.politics.guns",
    "19997_talk.politics.mideast",
    "19997_talk.religion.misc",
    "19997_alt.atheism",
    "19997_soc.religion.christian",
]

newsgroup_meaningful_labels = [
    "computer graphics",
    "computer operation system microsoft windows",
    "computer system ibm pc hardware",
    "computer system mac hardware",
    "windows x computer",
    "autos recommendation",
    "motorcycles recommendation",
    "baseball sport recommendation",
    "hockey sport recommendation",
    "crypt science",
    "electronics science",
    "medical science",
    "space science",
    "miscellaneous for sale",
    "miscellaneous in politics talk",
    "politics talk about guns",
    "mideast politics talk",
    "miscellaneous religion talk",
    "athletic atheism",
    "christian religion society",
]  # Extend the abbreviations of classes_20newsgroup.

TACRED_SEMANTIC_LABEL = {
    'org:founded_by': 'organization related: founded by', 'per:employee_of': 'person related: employee of',
    'org:alternate_names': 'organization related: alternate names',
    'per:cities_of_residence': 'person related: city of residence',
    'per:children': 'person related: children', 'per:title': 'person related: title',
    'per:siblings': 'person related: siblings', 'per:religion': 'person related: religion',
    'per:age': 'person related: age', 'org:website': 'organization related: website',
    'per:stateorprovinces_of_residence': 'person related: state or provinces of residence',
    'org:member_of': 'organization related: member of',
    'org:top_members/employees': 'organization related: top members/employees',
    'per:countries_of_residence': 'person related: countries of residence',
    'org:city_of_headquarters': 'organization related: city of headquarters',
    'org:members': 'organization related: members',
    'org:country_of_headquarters': 'organization related: country of headquarters',
    'per:spouse': 'person related: spouse',
    'org:stateorprovince_of_headquarters': 'organization related: state or province of headquarters',
    'org:number_of_employees/members': 'organization related: number of employees/members',
    'org:parents': 'organization related: parents', 'org:subsidiaries': 'organization related: subsidiaries',
    'per:origin': 'person related: origin',
    'org:political/religious_affiliation': 'organization related: political/religious affiliation',
    'per:other_family': 'person related: other family',
    'per:stateorprovince_of_birth': 'person related: state or province of birth',
    'org:dissolved': 'organization related: dissolved', 'per:date_of_death': 'person related: date of death',
    'org:shareholders': 'organization related: shareholders', 'per:alternate_names': 'person related: alternate names',
    'per:parents': 'person related: parents', 'per:schools_attended': 'person related: schools attended',
    'per:cause_of_death': 'person related: cause of death', 'per:city_of_death': 'person related: city of death',
    'per:stateorprovince_of_death': 'person related: state or province of death',
    'org:founded': 'organization related: founded', 'per:country_of_birth': 'person related: country of birth',
    'per:date_of_birth': 'person related: date of birth', 'per:city_of_birth': 'person related: city of birth',
    'per:charges': 'person related: charges'
}  # Expand the abbreviation in the original labels.

# For relation classification, we make sure the augmented input text also contains two entities.
FEWREL_LABEL_MAP = {
    'place served by transport hub': '[E11] place name [E12] place served by transport hub [E21] hub name [E22]',
    'mountain range': '[E11] geographical item name [E12] mountain range [E21] mountain range name [E22]',
    'religion': '[E11] item name [E12] religion [E21] religion name [E22]',
    'participating team': '[E11] team name [E12] participating team [E21] match or race name [E22]',
    'contains administrative territorial entity': '[E11] administrative territorial entity name [E12] contains administrative territorial entity [E21] direct subdivision name [E22]',
    'head of government': '[E11] government name [E12] head of government [E21] leader name [E22]',
    'country of citizenship': '[E11] person name [E12] country of citizenship [E21] country name [E22]',
    'original network': '[E11] radio or television name [E12] original network [E21] network name [E22]',
    'heritage designation': '[E11] cultural or national site name [E12] heritage designation [E21] heritage designation name [E22]',
    'performer': '[E11] art work name [E12] performer [E21] performer name [E22]',
    'participant of': '[E11] person/organization name [E12] participant of [E21] event name [E22]',
    'position held': '[E11] person name [E12] position held [E21] object position / public office [E22]',
    'has part': '[E11] entity name [E12] has part [E21] part of this subject name [E22]',
    'location of formation': '[E11] group/organization name [E12] location of formation [E21] location name [E22]',
    'located on terrain feature': '[E11] entity name [E12] located on terrain feature [E21] landform name [E22]',
    'architect': '[E11] building name [E12] architect [E21] person or architectural firm name',
    'country of origin': '[E11] entity name [E12] country of origin [E21] country name [E22]',
    'publisher': '[E11] entity name [E12] publisher [E21] person/organization name [E22]',
    'director': '[E11] entity name [E12] director [E21] person name [E22]',
    'father': '[E11] person name [E12] father [E21] father name [E22]',
    'developer': '[E11] entity name [E12] developer [E21] person/organization name [E22]',
    'military branch': '[E11] military unit, award, office, or person [E12] military branch [E21] branch name [E22]',
    'mouth of the watercourse': '[E11] water name [E12] mouth of the watercourse [E21] water name [E22]',
    'nominated for': '[E11] person, organisation or creative work [E12] nominated for [E21] award name [E22]',
    'movement': '[E11] person or work name [E12] movement [E21] movement name [E22]',
    'successful candidate': '[E11] election name [E12] successful candidate [E21] person name [E22]',
    'followed by': '[E11] subject name [E12] followed by [E21] subject name [E22]',
    'manufacturer': '[E11] product name [E12] manufacturer [E21] person/organization name [E22]',
    'instance of': '[E11] entity name [E12] instance of [E21] class name [E22]',
    'after a work by': '[E11] work name [E12] after a work by [E21] work name [E22]',
    'member of political party': '[E11] person name [E12] member of political party [E21] party name [E22]',
    'licensed to broadcast to': '[E11] station name [E12] licensed to broadcast to [E21] place name [E22]',
    'headquarters location': '[E11] organization name [E12] headquarters location [E21] place name [E22]',
    'sibling': '[E11] person name [E12] sibling [E21] sibling name [E22]',
    'instrument': '[E11] person name [E12] play instrument [E21] instrument name [E22]',
    'country': '[E11] person name [E12] country [E21] country name [E22]',
    'occupation': '[E11] person name [E12] has occupation [E21] occupation name [E22]',
    'residence': '[E11] person name [E12] residence [E21] location [E22]',
    'work location': '[E11] person name [E12] work location [E21] location [E22]',
    'subsidiary': '[E11] organization name [E12] subsidiary [E21] smaller organization name [E22]',
    'participant': '[E11] event name [E12] participant [E21] person name [E22]',
    'operator': '[E11] equipment name [E12] operator [E21] person, profession, or organization [E22]',
    'characters': '[E11] item name [E12] characters [E21] person name [E22]',
    'occupant': '[E11] person/organization name [E12] occupant [E21] property [E22]',
    'genre': '[E11] work/artist [E12] genre [E21] field or work [E22]',
    'operating system': '[E11] device/software [E12] operating system [E21] system name [E22]',
    'owned by': '[E11] item name [E12] owned by [E21] person name [E22]',
    'platform': '[E11] work name [E12] platform [E21] platform name [E22]',
    'tributary': '[E11] river name [E12] tributary [E21] river name [E22]',
    'winner': '[E11] event name [E12] winner [E21] person name [E22]',
    'said to be the same as': '[E11] entity name [E12] said to be the same as [E21] entity name [E22]',
    'composer': '[E11] music name [E12] composer [E21] person name [E22]',
    'league': '[E11] team/player name [E12] league [E21] league name [E22]',
    'record label': '[E11] music name [E12] record label [E21] band/trademark [E22]',
    'distributor': '[E11] work name [E12] distributor [E21] distributor name [E22]',
    'screenwriter': '[E11] work name [E12] screenwriter [E21] person name [E22]',
    'sports season of league or competition': '[E11] year [E12] sports season of league or competition [E21] season name [E22]',
    'taxon rank': '[E11] entity name [E12] taxonomy rank [E21] hierarchy [E22]',
    'location': '[E11] entity name [E12] location [E21] place [E22]',
    'field of work': '[E11] person/organization name [E12] field or work [E21] specialization [E22]',
    'language of work or name': '[E11] work name [E12] language or work or name [E21] language [E22]',
    'applies to jurisdiction': '[E11] item name [E12] applies to jurisdiction [E21] territorial jurisdiction [E22]',
    'notable work': '[E11] person name [E12] notable work [E21] work name [E22]',
    'located in the administrative territorial entity': '[E11] item name [E12] located in the administrative territorial entity [E21] location [E22]',
    'crosses': '[E11] bridge name [E12] crosses [E21] obstacle name [E22]',
    'original language of film or TV show': '[E11] work name [E12] original language of film or TV show [E21] language name [E22]',
    'competition class': '[E11] participant name [E12] competition class [E21] match name [E22]',
    'part of': '[E11] entity name [E12] is part of [E21] entity name [E22]',
    'sport': '[E11] participant name [E12] sport [E21] sport name [E22]',
    'constellation': '[E11] star name [E12] cnstellation [E21] constellation name [E22]',
    'position played on team / speciality': '[E11] person name [E12] position played on team / speciality [E21] position name [E22]',
    'located in or next to body of water': '[E11] place name [E12] located in or next to body of water [E21] water name [E22]',
    'voice type': '[E11] person name [E12] voice type [E21] voice type [E22]',
    'follows': '[E21] entity [E22] follows [E11] entity [E12]',
    'spouse': '[E11] person name [E12] spouse [E21] spouse name [E22]',
    'military rank': '[E11] person name [E12] military rank [E21] rank name [E22]',
    'mother': '[E11] person name [E12] mother [E21] mother name [E22]',
    'member of': '[E11] person name [E12] member of [E21] organization name [E22]',
    'child': '[E11] person name [E12] child [E21] child name [E22]',
    'main subject': '[E11] work name [E12] main subject [E21] topic name [E22]'
}

TACRED_LABEL_MAP = {
    'organization related: founded by': '[E11] organization name [E12] founded by [E21] person name [E22]',
    'person related: employee of': '[E11] person name [E12] employee of [E21] company name [E22]',
    'organization related: alternate names': '[E11] organization name [E12] alternate names [E21] organization alternate name [E22]',
    'person related: city of residence': '[E11] person name [E12] city of residence [E21] city name [E22]',
    'person related: children': '[E11] person name [E12] children [E21] child name [E22]',
    'person related: title': '[E11] person name [E12] title [E21] title name [E22]',
    'person related: siblings': '[E11] person name [E12] siblings [E21] sibling name [E22]',
    'person related: religion': '[E11] person name [E12] religion [E21] religion name [E22]',
    'person related: age': '[E11] person name [E12] age [E21] age number [E22]',
    'organization related: website': '[E11] organization name [E12] website [E21] website url [E22]',
    'person related: state or provinces of residence': '[E11] person name [E12] state or provinces of residence [E21] state or province name [E22]',
    'organization related: member of': '[E11] organization name [E12] member of [E21] larger organization name [E22]',
    'organization related: top members/employees': '[E11] organization name [E12] top members/employees [E21] person name [E22]',
    'person related: countries of residence': '[E11] person name [E12] countries of residence [E21] country name [E22]',
    'organization related: city of headquarters': '[E11] organization name [E12] city of headquarters [E21] city name [E22]',
    'organization related: members': '[E11] organization name [E12] members [E21] member name [E22]',
    'organization related: country of headquarters': '[E11] organization name [E12] country of headquarters [E21] country name [E22]',
    'person related: spouse': '[E11] person name [E12] spouse [E21] spouse name [E22]',
    'organization related: state or province of headquarters': '[E11] organization name [E12] state or province of headquarters [E21] state or province name [E22]',
    'organization related: number of employees/members': '[E11] organization name [E12] number of employees/members [E21] number [E22]',
    'organization related: parents': '[E11] organization name [E12] parents [E21] parent name [E22]',
    'organization related: subsidiaries': '[E11] organization name [E12] subsidiaries [E21] subsidiary nae [E22]',
    'person related: origin': '[E11] person name [E12] origin [E21] country name [E22]',
    'organization related: political/religious affiliation': '[E11] organization name [E12] political/religious affiliation [E21] affiliation name [E22]',
    'person related: other family': '[E11] person name [E12] other family [E21] person name [E22]',
    'person related: state or province of birth': '[E11] person name [E12] state or province of birth [E21] state or province name [E22]',
    'organization related: dissolved': '[E11] organization name [E12] dissolved [E21] time [E22]',
    'person related: date of death': '[E11] person name [E12] date of death [E21] date [E22]',
    'organization related: shareholders': '[E11] organization name [E12] shareholders [E21] organization name [E22]',
    'person related: alternate names': '[E11] person name [E12] alternate names [E21] person alternate name [E22]',
    'person related: parents': '[E11] person name [E12] parents [E21] parent name [E22]',
    'person related: schools attended': '[E11] person name [E12] schools attended [E21] school name [E22]',
    'person related: cause of death': '[E11] person name [E12] cause of death [E21] cause [E22]',
    'person related: city of death': '[E11] person name [E12] city of death [E21] city name [E22]',
    'person related: state or province of death': '[E11] person name [E12] state or province of death [E21] state or province name [E22]',
    'organization related: founded': '[E11] organization name [E12] founded [E21] time [E22]',
    'person related: country of birth': '[E11] person name [E12] country of birth [E21] country name [E22]',
    'person related: date of birth': '[E11] person name [E12] date of birth [E21] date [E22]',
    'person related: city of birth': '[E11] person name [E12] city of birth [E21] city name [E22]',
    'person related: charges': '[E11] person name [E12] charges [E21] crime name [E22]'
}

DATA_ROOT = './data'


def get_dataset(dataset_name, tokenizer, args, return_label_set=False):
    """Returns:
        {
            task_id: {'train': Dataset('text', 'semantic_labels', 'labels'),
                        'dev': Dataset('text', 'semantic_labels', 'labels'),
                        'test':Dataset('text', 'semantic_labels', 'labels')}
        }
    """
    if 'banking77' in dataset_name:
        data = {k: {'train': {'text': [], 'semantic_labels': [], 'labels': []},
                    'test': {'text': [], 'semantic_labels': [], 'labels': []}}
                for k in range(7)}  # 7 groups, 10 classes per group.
        label_names = [label.replace('_', ' ') for label in BANKING_LABEL2ID.keys()]
        label_set = {k: {idx: label_names[idx] for idx in range(k * 10, (k + 1) * 10)} for k in range(7)}
        train_df = pd.read_csv(os.path.join(DATA_ROOT, 'banking77/train.csv'))
        for i in range(len(train_df)):
            label_id = BANKING_LABEL2ID[train_df.at[i, 'category']]
            group = label_id // 10
            if group == 7:
                continue
            data[group]['train']['text'].append(train_df.at[i, 'text'])
            data[group]['train']['semantic_labels'].append(train_df.at[i, 'category'].replace('_', ' '))
            data[group]['train']['labels'].append(label_id)
        test_df = pd.read_csv(os.path.join(DATA_ROOT, 'banking77/test.csv'))
        for i in range(len(test_df)):
            label_id = BANKING_LABEL2ID[test_df.at[i, 'category']]
            group = label_id // 10
            if group == 7:
                continue
            data[group]['test']['text'].append(test_df.at[i, 'text'])
            data[group]['test']['semantic_labels'].append(test_df.at[i, 'category'].replace('_', ' '))
            data[group]['test']['labels'].append(label_id)
        for k in data:
            # We split 20% training data as the validation set.
            data[k]['train'] = Dataset.from_dict(data[k]['train'])
            train_dev = data[k]['train'].train_test_split(test_size=0.2, seed=2022, shuffle=True)
            data[k]['train'] = train_dev['train']
            data[k]['dev'] = train_dev['test']
            data[k]['test'] = Dataset.from_dict(data[k]['test'])

    elif 'clinc150' in dataset_name:
        with open(os.path.join(DATA_ROOT, 'clinc150/data_full.json'), 'r') as f:
            raw_data = json.load(f)
        with open(os.path.join(DATA_ROOT, 'clinc150/label_dict.json'), 'r') as f:
            label_dict = json.load(f)

        data = {k: {'train': {'text': [], 'semantic_labels': [], 'labels': []},
                    'test': {'text': [], 'semantic_labels': [], 'labels': []},
                    'dev': {'text': [], 'semantic_labels': [], 'labels': []}}
                for k in range(15)}  # 15 groups, 10 classes per group.
        label_names = [label.replace('_', ' ') for label in label_dict.keys()][:150]  # We don't consider the 'oos'.
        label_set = {k: {idx: label_names[idx] for idx in range(k * 10, (k + 1) * 10)} for k in range(15)}

        for ds in ['train', 'test', 'dev']:
            new_ds = 'val' if ds == 'dev' else ds
            for item in raw_data[new_ds]:
                label_id = label_dict[item[1]]
                group = label_id // 10
                data[group][ds]['text'].append(item[0])
                data[group][ds]['semantic_labels'].append(item[1].replace('_', ' '))
                data[group][ds]['labels'].append(label_id)
        for k in data:
            data[k]['train'] = Dataset.from_dict(data[k]['train'])
            data[k]['test'] = Dataset.from_dict(data[k]['test'])
            data[k]['dev'] = Dataset.from_dict(data[k]['dev'])

    elif '20news' in dataset_name:
        # http://qwone.com/~jason/20Newsgroups/, we use the "bydate" version.
        # 2 classes per task
        label_names = newsgroup_meaningful_labels
        label_set = {k: {idx: label_names[idx] for idx in range(k * 2, (k + 1) * 2)} for k in range(10)}
        train_per_class = 500

        processed_data_file = os.path.join(DATA_ROOT,
                                           f'20news_2classes_per_task.json')
        if os.path.exists(processed_data_file):
            data = load_json(processed_data_file)
            data = {int(k): v for k, v in data.items()}
        else:
            data = {k: {'train': {'text': [], 'semantic_labels': [], 'labels': []},
                        'test': {'text': [], 'semantic_labels': [], 'labels': []}}
                    for k in range(10)}  # 10 groups, 2 classes per group.
            for i in range(20):
                group_id = i // 2
                d = load_dataset('newsgroup', classes_20newsgroup[i], split='train')
                for x in d['text'][:train_per_class]:
                    data[group_id]['train']['text'].append(x)
                    data[group_id]['train']['semantic_labels'].append(newsgroup_meaningful_labels[i])
                    data[group_id]['train']['labels'].append(i)
                for x in d['text'][train_per_class:]:
                    data[group_id]['test']['text'].append(x)
                    data[group_id]['test']['semantic_labels'].append(newsgroup_meaningful_labels[i])
                    data[group_id]['test']['labels'].append(i)
            os.makedirs(DATA_ROOT, exist_ok=True)
            dump_json(data, processed_data_file)

        for k in data:
            # For 20news, the split is train:dev:test = 5:2:3
            data[k]['train'] = Dataset.from_dict(data[k]['train'])
            data[k]['test'] = Dataset.from_dict(data[k]['test'])
            dev_test = data[k]['test'].train_test_split(test_size=0.6, seed=2022, shuffle=True)
            data[k]['dev'] = dev_test['train']
            data[k]['test'] = dev_test['test']

    elif 'fewrel' in dataset_name:
        with open(os.path.join(DATA_ROOT, 'FewRel-2021.pkl'), 'rb') as f:
            datas = pickle.load(f)
        train_dataset, val_dataset, test_dataset = datas
        dataset = {'train': train_dataset, 'dev': val_dataset, 'test': test_dataset}
        # Format of each sample:
        # {'relation': label, 'text': text, 'semantic_label': semantic label, 'label_explanation': label explanation}
        label_set = {k: {idx: train_dataset[idx][0]['semantic_label'] for idx in range(k * 10, (k + 1) * 10)} for k in
                     range(8)}
        data = {k: {'train': {'text': [], 'semantic_labels': [], 'labels': []},
                    'dev': {'text': [], 'semantic_labels': [], 'labels': []},
                    'test': {'text': [], 'semantic_labels': [], 'labels': []}}
                for k in range(8)}  # 8 groups, 10 classes per group.
        for split in ['train', 'dev', 'test']:
            for i in range(80):
                for sample in dataset[split][i]:
                    group_id = i // 10
                    data[group_id][split]['text'].append(sample['text'])
                    data[group_id][split]['semantic_labels'].append(sample['semantic_label'])
                    data[group_id][split]['labels'].append(i)
        for k in data:
            data[k]['train'] = Dataset.from_dict(data[k]['train'])
            data[k]['dev'] = Dataset.from_dict(data[k]['dev'])
            data[k]['test'] = Dataset.from_dict(data[k]['test'])

    elif 'tacred' in dataset_name:
        with open(os.path.join(DATA_ROOT, 'TACRED-2021.pkl'), 'rb') as f:
            datas = pickle.load(f)
        train_dataset, _, test_dataset = datas
        dataset = {'train': train_dataset, 'test': test_dataset}
        # Format of each sample:
        # {'relation': label, 'text': text, 'semantic_label': semantic label}
        label_set = {k: {idx: TACRED_SEMANTIC_LABEL[train_dataset[idx][0]['semantic_label']] for idx in
                         range(k * 5, (k + 1) * 5)} for k in range(8)}
        data = {k: {'train': {'text': [], 'semantic_labels': [], 'labels': []},
                    'dev': {'text': [], 'semantic_labels': [], 'labels': []},
                    'test': {'text': [], 'semantic_labels': [], 'labels': []}}
                for k in range(8)}  # 8 groups, 5 classes per group.

        for split in ['train', 'test']:
            for i in range(40):
                for sample in dataset[split][i]:
                    group_id = i // 5
                    data[group_id][split]['text'].append(sample['text'])
                    data[group_id][split]['semantic_labels'].append(TACRED_SEMANTIC_LABEL[sample['semantic_label']])
                    data[group_id][split]['labels'].append(i)
        for k in data:
            # We split 20% training data as the validation set.
            data[k]['train'] = Dataset.from_dict(data[k]['train'])
            train_dev = data[k]['train'].train_test_split(test_size=0.2, seed=2022, shuffle=True)
            data[k]['train'] = train_dev['train']
            data[k]['dev'] = train_dev['test']
            data[k]['test'] = Dataset.from_dict(data[k]['test'])

    else:
        raise NotImplementedError(f'{dataset_name} is not supported yet!')

    if return_label_set:
        return data, label_set
    else:
        return data
