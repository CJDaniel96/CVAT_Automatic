# coding: utf-8
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Table, Text, create_engine, text, func
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata
# engine_url = 'postgresql://postgres:postgres@10.0.4.188:5432/ai'
engine_url = 'postgresql://postgres:postgres@127.0.0.1:5432/ai'
engine = create_engine(engine_url, echo=True)


class AiModelInfo(Base):
    __tablename__ = 'ai_model_info'
    __table_args__ = {'schema': 'amr'}

    model_id = Column(Integer, primary_key=True, nullable=False, unique=True,
                      server_default=text("nextval('amr.ai_model_info_model_id_seq'::regclass)"))
    model_type = Column(String(20), nullable=False)
    model_name = Column(Text, nullable=False)
    model_path = Column(Text, nullable=False)
    model_version = Column(Text, nullable=False)
    comp_category = Column(Text, nullable=False)
    defect_category = Column(Text, nullable=False)
    verified_status = Column(String(20), nullable=False, server_default=text(
        "'UNVERIFIED'::character varying"))
    create_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    sched_id = Column(Integer)


class AiModelPerf(Base):
    __tablename__ = 'ai_model_perf'
    __table_args__ = {'schema': 'amr'}

    model_id = Column(Integer, primary_key=True, nullable=False, unique=True)
    metrics_result = Column(Text, nullable=False)
    false_negative_imgs = Column(Text, nullable=False)
    false_positive_imgs = Column(Text, nullable=False)
    insert_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class AiServerInfo(Base):
    __tablename__ = 'ai_server_info'
    __table_args__ = {'schema': 'amr'}

    server_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.ai_server_info_server_id_seq'::regclass)"))
    server_name = Column(String(30))
    server_type = Column(String(30))
    server_url = Column(Text)


class AmrModelSlot(Base):
    __tablename__ = 'amr_model_slot'
    __table_args__ = {'schema': 'amr'}

    slot_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.amr_model_slot_slot_id_seq'::regclass)"))
    comp_category = Column(String, nullable=False)
    defect_category = Column(String, nullable=False)
    model_func = Column(String, nullable=False)


class AmrPositionDeploy(Base):
    __tablename__ = 'amr_position_deploy'
    __table_args__ = {'schema': 'amr'}

    position_id = Column(ForeignKey(
        'amr.amr_position_info.position_id'), primary_key=True, nullable=False)
    slot_id = Column(ForeignKey('amr.amr_model_slot.slot_id'),
                     primary_key=True, nullable=False)
    model_id = Column(ForeignKey('amr.ai_model_info.model_id'),
                      primary_key=True, nullable=False)
    create_time = Column(DateTime, nullable=False,
                         server_default=text("CURRENT_TIMESTAMP"))

    model = relationship('AiModelInfo')
    position = relationship('AmrPositionInfo')
    slot = relationship('AmrModelSlot')


class AmrPositionInfo(Base):
    __tablename__ = 'amr_position_info'
    __table_args__ = {'schema': 'amr'}

    position_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.amr_position_info_position_id_seq'::regclass)"))
    site = Column(String)
    line = Column(String)


class AugImg(Base):
    __tablename__ = 'aug_img'
    __table_args__ = {'schema': 'amr'}

    aug_img_id = Column(BigInteger, primary_key=True, nullable=False, server_default=text(
        "nextval('amr.aug_img_aug_img_id_seq'::regclass)"))
    img_category_id = Column(Integer, primary_key=True, nullable=False)
    src_img_id = Column(BigInteger, primary_key=True, nullable=False)
    aug_method = Column(String(15))
    img_create_time = Column(DateTime(True), primary_key=True,
                             nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    insert_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class CropCategorizingRecord(Base):
    __tablename__ = 'crop_categorizing_record'
    __table_args__ = {'schema': 'amr'}

    finetune_id = Column(Integer, primary_key=True, nullable=False)
    ori_img_id = Column(Text, primary_key=True,
                        nullable=False, comment='原圖 ID')
    crop_name = Column(Text, primary_key=True, nullable=False)
    inspector_id = Column(Integer, nullable=False,
                          comment='代碼:1 = ai, 2= human')
    x_min = Column(Float(53), nullable=False)
    y_min = Column(Float(53), nullable=False)
    x_max = Column(Float(53), nullable=False)
    y_max = Column(Float(53), nullable=False)
    categorizing_code = Column(String(10), nullable=False, server_default=text(
        "'DEFAULT'::character varying"), comment='分類結果')
    update_time = Column(DateTime(True), nullable=False, server_default=text(
        "CURRENT_TIMESTAMP"), comment='更新時間')
    create_time = Column(DateTime(True), nullable=False, server_default=text(
        "CURRENT_TIMESTAMP"), comment='建立時間')


class CropImg(Base):
    __tablename__ = 'crop_img'
    __table_args__ = {'schema': 'amr'}

    crop_img_id = Column(BigInteger, primary_key=True, nullable=False, server_default=text(
        "nextval('amr.crop_img_crop_img_id_seq'::regclass)"))
    src_img_id = Column(BigInteger, primary_key=True, nullable=False)
    img_length = Column(Integer)
    img_width = Column(Integer)
    img_create_time = Column(DateTime(True), primary_key=True,
                             nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    insert_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class DefectCategoryInfo(Base):
    __tablename__ = 'defect_category_info'
    __table_args__ = {'schema': 'amr'}

    defect_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.defect_category_info_defect_id_seq'::regclass)"))
    detection_code = Column(String(10))
    defect_category = Column(String(20))
    defect_type = Column(Text)


class DetectionRecord(Base):
    __tablename__ = 'detection_record'
    __table_args__ = {'schema': 'amr'}

    img_category_id = Column(Integer, primary_key=True, nullable=False)
    src_img_id = Column(BigInteger, primary_key=True, nullable=False)
    inspector_id = Column(Integer, primary_key=True, nullable=False)
    defect_category = Column(String(30), primary_key=True, nullable=False)
    detection_code = Column(String(10), nullable=False,
                            server_default=text("'DEFAULT'::character varying"))
    inspect_time = Column(DateTime(True), primary_key=True,
                          nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    insert_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class ImgCategoryInfo(Base):
    __tablename__ = 'img_category_info'
    __table_args__ = {'schema': 'amr'}

    img_type_id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.img_category_info_img_type_id_seq'::regclass)"))
    img_category = Column(String(15))


class ImgPath(Base):
    __tablename__ = 'img_path'
    __table_args__ = {'schema': 'amr'}

    img_category_id = Column(Integer, primary_key=True, nullable=False)
    src_img_id = Column(BigInteger, primary_key=True, nullable=False)
    server_id = Column(Integer, primary_key=True, nullable=False)
    img_file_path = Column(Text)
    create_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class InspectorInfo(Base):
    __tablename__ = 'inspector_info'
    __table_args__ = {'schema': 'amr'}

    inspector_id = Column(Integer, primary_key=True, nullable=False, server_default=text(
        "nextval('amr.inspector_info_inspector_id_seq'::regclass)"))
    inspector_type = Column(String(10))
    inspector_name = Column(String(30))
    create_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class IriRecord(Base):
    __tablename__ = 'iri_record'
    __table_args__ = {'schema': 'amr'}

    id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.iri_record_id_seq'::regclass)"), comment='No.')
    project = Column(Text)
    task = Column(Text)
    status = Column(Text)
    site = Column(String)
    line = Column(String)
    type = Column(String)
    _from = Column('from', DateTime(True), comment='Time From')
    to = Column(DateTime(True), comment='Time To')
    labeling = Column(Boolean)
    od_training = Column(Text)
    cls_training = Column(Text)
    update_time = Column(DateTime(True), onupdate=func.now())
    create_time = Column(DateTime(True), nullable=False,
                         server_default=text("CURRENT_TIMESTAMP"))
    task_id = Column(Integer)


class OdTrainingInfo(Base):
    __tablename__ = 'od_training_info'
    __table_args__ = {'schema': 'amr'}

    id = Column(Integer, primary_key=True, server_default=text(
        "nextval('amr.od_traning_info_id_seq'::regclass)"))
    datetime = Column(DateTime(True), nullable=False)
    query_id = Column(Integer, nullable=False)
    task_id = Column(Integer, nullable=False)
    comp_type = Column(String(32), nullable=False)
    val_status = Column(String(32), nullable=False)
    model_version = Column(String(20))


class OriginImg(Base):
    __tablename__ = 'origin_img'
    __table_args__ = {'schema': 'amr'}

    img_id = Column(BigInteger, primary_key=True, nullable=False, server_default=text(
        "nextval('amr.origin_img_img_id_seq'::regclass)"))
    site = Column(String(5), primary_key=True, nullable=False)
    line = Column(String(10), primary_key=True, nullable=False)
    product_name = Column(String(20))
    carrier_code = Column(String)
    comp_category = Column(String)
    comp_type = Column(String)
    comp_name = Column(String)
    light_type = Column(String)
    img_create_time = Column(DateTime(True), primary_key=True,
                             nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    insert_time = Column(DateTime(True), primary_key=True,
                         nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class TrainingSchedule(Base):
    __tablename__ = 'training_schedule'
    __table_args__ = {'schema': 'amr'}

    sched_id = Column(Integer, primary_key=True, nullable=False, server_default=text(
        "nextval('amr.training_schedule_sched_id_seq'::regclass)"))
    training_settings = Column(Text, nullable=False)
    dataset_request = Column(Text, nullable=False)
    sched_status = Column(String(20), nullable=False,
                          server_default=text("'Waiting'::character varying"))
    sched_insert_time = Column(DateTime(
        True), primary_key=True, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    training_start = Column(DateTime(True))
    training_end = Column(DateTime(True))


t_v_amr_deploy_overview = Table(
    'v_amr_deploy_overview', metadata,
    Column('position_id', Integer),
    Column('slot_id', Integer),
    Column('model_id', Integer),
    Column('deploy_time', DateTime),
    Column('site', String),
    Column('line', String),
    Column('slot_comp_category', String),
    Column('slot_defect_category', String),
    Column('model_func', String),
    Column('model_name', Text),
    Column('model_version', Text),
    Column('model_path', Text),
    Column('model_comp_category', Text),
    Column('model_defect_category', Text),
    schema='amr'
)


t_v_amr_detection_record_origin = Table(
    'v_amr_detection_record_origin', metadata,
    Column('img_category_id', Integer),
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    schema='amr'
)


t_v_detection_record_crop = Table(
    'v_detection_record_crop', metadata,
    Column('img_category', String(15)),
    Column('crop_img_id', BigInteger),
    Column('src_img_id', BigInteger),
    Column('img_length', Integer),
    Column('img_width', Integer),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    Column('inspector_type', String(10)),
    Column('inspector_name', String(30)),
    Column('inspect_time', DateTime(True)),
    schema='amr'
)


t_v_detection_record_finetune_crop = Table(
    'v_detection_record_finetune_crop', metadata,
    Column('img_category', String(15)),
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('crop_img_id', BigInteger),
    Column('img_length', Integer),
    Column('img_width', Integer),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    Column('inspector_type', String(10)),
    Column('inspector_name', String(30)),
    Column('inspect_time', DateTime(True)),
    schema='amr'
)


t_v_detection_record_finetune_origin = Table(
    'v_detection_record_finetune_origin', metadata,
    Column('img_category', String(15)),
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    Column('inspector_type', String(10)),
    Column('inspector_name', String(30)),
    Column('inspect_time', DateTime(True)),
    schema='amr'
)


t_v_detection_record_meta = Table(
    'v_detection_record_meta', metadata,
    Column('img_category', String(15)),
    Column('inspector_name', String(30)),
    Column('inspector_type', String(10)),
    Column('img_category_id', Integer),
    Column('src_img_id', BigInteger),
    Column('defect_category', String(30)),
    Column('detection_code', String(10)),
    Column('inspect_time', DateTime(True)),
    schema='amr'
)


t_v_detection_record_origin = Table(
    'v_detection_record_origin', metadata,
    Column('img_category', String(15)),
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    Column('inspector_type', String(10)),
    Column('inspector_name', String(30)),
    Column('inspect_time', DateTime(True)),
    schema='amr'
)


t_v_finetune_dataset_aug = Table(
    'v_finetune_dataset_aug', metadata,
    Column('aug_img_id', BigInteger),
    Column('aug_method', String(15)),
    Column('src_img_id', BigInteger),
    Column('img_file_path', Text),
    schema='amr'
)


t_v_finetune_dataset_crop = Table(
    'v_finetune_dataset_crop', metadata,
    Column('img_category', String(15)),
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('crop_img_id', BigInteger),
    Column('img_length', Integer),
    Column('img_width', Integer),
    Column('detection_code', String(10)),
    Column('defect_category', String(30)),
    Column('inspector_type', String(10)),
    Column('inspector_name', String(30)),
    Column('inspect_time', DateTime(True)),
    Column('img_file_path', Text),
    schema='amr'
)


t_v_img_path_aug = Table(
    'v_img_path_aug', metadata,
    Column('aug_img_id', BigInteger),
    Column('img_category_id', Integer),
    Column('src_img_id', BigInteger),
    Column('aug_method', String(15)),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('server_name', String(30)),
    Column('server_type', String(30)),
    Column('server_url', Text),
    Column('img_file_path', Text),
    schema='amr'
)


t_v_img_path_crop = Table(
    'v_img_path_crop', metadata,
    Column('crop_img_id', BigInteger),
    Column('src_img_id', BigInteger),
    Column('img_length', Integer),
    Column('img_width', Integer),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('server_name', String(30)),
    Column('server_type', String(30)),
    Column('server_url', Text),
    Column('img_file_path', Text),
    schema='amr'
)


t_v_img_path_origin = Table(
    'v_img_path_origin', metadata,
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('server_name', String(30)),
    Column('server_type', String(30)),
    Column('server_url', Text),
    Column('img_file_path', Text),
    schema='amr'
)


t_v_origin_crop_img_pair = Table(
    'v_origin_crop_img_pair', metadata,
    Column('img_id', BigInteger),
    Column('site', String(5)),
    Column('line', String(10)),
    Column('product_name', String(20)),
    Column('carrier_code', String),
    Column('comp_category', String),
    Column('comp_type', String),
    Column('comp_name', String),
    Column('light_type', String),
    Column('img_create_time', DateTime(True)),
    Column('insert_time', DateTime(True)),
    Column('crop_img_id', BigInteger),
    Column('img_length', Integer),
    Column('img_width', Integer),
    schema='amr'
)


def create_session():
    Session = sessionmaker(bind=engine)
    session = Session()

    return session
