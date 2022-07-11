from models.ai_models import IriRecord, create_session as ai_create_session
from models.cvat_models import create_session as cvat_create_session


if __name__ == '__main__':
    cnt = 0
    session = ai_create_session()
    while True:
        task = session.query(IriRecord).filter(IriRecord.status == 'status').order_by(IriRecord.update_time).first()
        if (task != None) or (cnt > 100):
            break
        else:
            cnt += 1
    session.close()