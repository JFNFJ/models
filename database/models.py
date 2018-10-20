from flask_sqlalchemy import SQLAlchemy
from settings import app

import datetime

db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String, nullable=False, unique=True)
    email = db.Column(db.String, nullable=False)
    password = db.Column(db.String, nullable=False)
    confirmed = db.Column(db.Boolean, nullable=False, default=False)

    def __init__(self, name, password, email):
        self.name = name
        self.password = bcrypt.encrypt(password)
        self.email = email

    topics = db.relationship("Topic", back_populates="user", cascade="all,delete")

    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}', password='{self.password}')>"


class Topic(db.Model):
    __tablename__ = "topics"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    name = db.Column(db.String)
    deadline = db.Column(db.Date)
    language = db.Column(db.String)

    user = db.relationship("User", back_populates="topics")
    general_result = db.relationship("GeneralResult", uselist=False, backref="topic", cascade="all,delete")
    evolution_results = db.relationship("EvolutionResult", back_populates="topic", cascade="all,delete")
    location_results = db.relationship("LocationResult", back_populates="topic", cascade="all,delete")
    source_results = db.relationship("SourceResult", back_populates="topic", cascade="all,delete")

    def __repr__(self):
        return f"<Topic(name='{self.name}', deadline='{self.deadline}', owner='{self.user_id}')>"


class Result:
    def increment_positive(self):
        self.positive += 1
        db.session.add(self)
        db.session.commit()

    def increment_neutral(self):
        self.neutral += 1
        db.session.add(self)
        db.session.commit()

    def increment_negative(self):
        self.negative += 1
        db.session.add(self)
        db.session.commit()


class GeneralResult(db.Model, Result):
    __tablename__ = "general_results"

    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), primary_key=True)
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)

    @staticmethod
    def increment_field(topic_id, field):
        result = GeneralResult.query \
            .filter(GeneralResult.topic_id == topic_id) \
            .one()
        increment_method = getattr(result, field)
        increment_method()
        return result

    def __repr__(self):
        return f"<GeneralResult(topic='{self.topic}', positive='{self.positive}', " \
               f"negative='{self.negative}', neutral='{self.neutral}')>"


class EvolutionResult(db.Model, Result):
    __tablename__ = "evolution_results"

    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), primary_key=True)
    day = db.Column(db.Date, primary_key=True)
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)

    topic = db.relationship("Topic", back_populates="evolution_results")

    @staticmethod
    def increment_field(topic_id, day, field):
        result = EvolutionResult.query \
            .filter(EvolutionResult.topic_id == topic_id) \
            .filter(EvolutionResult.day == day)\
            .one()
        increment_method = getattr(result, field)
        increment_method()
        return result

    def __repr__(self):
        return f"<EvolutionResult(topic='{self.topic}', positive='{self.positive}', " \
               f"negative='{self.negative}', neutral='{self.neutral}', day='{self.day}')>"


class LocationResult(db.Model, Result):
    __tablename__ = "location_results"

    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), primary_key=True)
    location = db.Column(db.String, primary_key=True)
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)

    topic = db.relationship("Topic", back_populates="location_results")

    @staticmethod
    def increment_field(topic_id, location, field):
        result = LocationResult.query \
            .filter(LocationResult.topic_id == topic_id) \
            .filter(LocationResult.location == location)\
            .one()
        increment_method = getattr(result, field)
        increment_method()
        return result

    def __repr__(self):
        return f"<EvolutionResult(topic='{self.topic}', positive='{self.positive}', " \
               f"negative='{self.negative}', neutral='{self.neutral}', location='{self.location}')>"

class SourceResult(db.Model, Result):
    __tablename__ = "source_results"

    topic_id = db.Column(db.Integer, db.ForeignKey('topics.id'), primary_key=True)
    source = db.Column(db.String, primary_key=True)
    positive = db.Column(db.Integer)
    negative = db.Column(db.Integer)
    neutral = db.Column(db.Integer)

    topic = db.relationship("Topic", back_populates="source_results")

    @staticmethod
    def increment_field(topic_id, source, field):
        result = SourceResult.query \
            .filter(SourceResult.topic_id == topic_id) \
            .filter(SourceResult.source == source)\
            .one()
        increment_method = getattr(result, field)
        increment_method()
        return result

    def __repr__(self):
        return f"<SourceResult(topic='{self.topic}', positive='{self.positive}', " \
               f"negative='{self.negative}', neutral='{self.neutral}', source='{self.source}')>"

    def to_dict(self):
        return {
            'topic_id': self.topic_id,
            'positive': self.positive,
            'negative': self.negative,
            'neutral': self.neutral,
            'source': self.source
        }
