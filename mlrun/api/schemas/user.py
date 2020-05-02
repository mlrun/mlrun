from pydantic import BaseModel


# Shared properties
class UserBase(BaseModel):
    name: str


# Properties to receive via API on creation
class UserCreate(UserBase):
    pass


# Properties to receive via API on update
class UserUpdate(UserBase):
    pass


class UserInDB(UserBase):
    id: int = None

    class Config:
        orm_mode = True


# Additional properties to return via API
class User(UserInDB):
    pass
