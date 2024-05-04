"""A library for creating and manipulating dataframes in a tabular format.
Basically a kinkier and a (much) slower version of pandas used for all the data science projects so that I can keep my
promise of building a machine leaning library without a requirements.txt file."""
import random


class Column:
    def __init__(self, name: str):
        self._name = name
        self._data = []
        self._type = "str"

    @property
    def data(self) -> list:
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, data_type: str):
        if self._type == data_type:
            return
        match data_type:
            case "int":
                for i in range(len(self._data)):
                    self._data[i] = int(self._data[i])
            case "float":
                for i in range(len(self._data)):
                    self._data[i] = float(self._data[i])
            case "str":
                pass
            case _:
                raise ValueError("Invalid data type")
        self._type = data_type

    def add_data(self, data):
        self._data.append(data)

    def count_occurrences(self, value_to_count):
        count = 0
        for value in self._data:
            if value == value_to_count:
                count += 1
        return count

    def append_column(self, other: 'Column'):
        for i in range(len(other)):
            self._data.append(other.data[i])

    def dot_product(self, other) -> float | int:
        if len(self) != len(other):
            raise ValueError("Columns must have the same length")

        if isinstance(other, Column):
            if not all(isinstance(x, (int, float, bool)) for x in self.data) or not all(
                    isinstance(y, (int, float, bool)) for y in other.data):
                raise ValueError("Columns must be entirely numeric")
            return sum(x * y for x, y in zip(self.data, other.data))

        if isinstance(other, list):
            if not all(isinstance(x, (int, float, bool)) for x in self.data) or not all(
                    isinstance(y, (int, float, bool)) for y in other):
                raise ValueError("Columns must be entirely numeric")
            return sum(x * y for x, y in zip(self.data, other))

    def __len__(self):
        return len(self._data)

    def __mul__(self, other):
        """Overloaded * operator for multiplying a column by a scalar or another column.
        If other is an instance of Column, the dot product is calculated and returned"""
        if isinstance(other, int) or isinstance(other, float):
            for i in range(len(self._data)):
                self._data[i] *= other
        elif isinstance(other, Column) or isinstance(other, list):
            return self.dot_product(other)

    def __rmul__(self, other):
        """Right multiplication operator"""
        return self.__mul__(other)

    def __str__(self):
        return f"<Column {self.name}; {self.type}; {len(self)}> | {self._data[:10]}"

    def __repr__(self):
        return f"<Column '{self._name}' with {len(self._data)} rows>"

    def __getitem__(self, item: int):
        return self._data[item]

    def __setitem__(self, key: int, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)


class Row:
    """Proxy class only used internally by the DataFrame class"""

    def __init__(self, dataframe, row_index):
        self._dataframe = dataframe
        self._row_index = row_index

    def __getitem__(self, column_index):
        return self._dataframe._data[column_index].data[self._row_index]

    def __setitem__(self, column_index, value):
        self._dataframe._data[column_index].data[self._row_index] = value

    def __len__(self):
        return self._dataframe._columns

    def __iter__(self):
        return (self._dataframe._data[col].data[self._row_index] for col in range(self._dataframe._columns))

    def __repr__(self):
        return str([self.__getitem__(col) for col in range(self._dataframe._columns)])


class DataFrame:
    def __init__(self, source=None):
        if source is None:
            source = {}
        self._input_data = source
        self._data = []  # list of Column objects
        self._rows = 0  # read only
        self._columns = 0  # read only
        if source:
            self.__extract_data_from_dict()

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def columns(self) -> int:
        return self._columns

    @rows.setter
    def rows(self, rows: int):
        raise AttributeError("rows is read only")

    @columns.setter
    def columns(self, columns: int):
        raise AttributeError("columns is read only")

    def column(self, name: str) -> Column:
        for column in self._data:
            if column.name == name:
                return column
        raise ValueError("Column not found")

    def __extract_data_from_dict(self):
        if len(self._input_data) == 0:
            return  # empty dictionary
        for key, value in self._input_data.items():
            column_obj = Column(key)
            column_obj.data = value
            self._data.append(column_obj)
        self._rows = len(self._data[0].data)
        self._columns = len(self._data)

    def __repr__(self):
        return self.__str__()

    def type(self, data_type: str):
        match data_type:
            case "int":
                for i in range(len(self._data)):
                    self._data[i].type = data_type
            case "float":
                for column in self._data:
                    column.type = data_type
            case "str":
                for column in self._data:
                    column.type = data_type
            case _:
                raise ValueError("Invalid data type")

    def shape(self) -> tuple:
        """Returns a tuple of (rows, columns)"""
        return self._rows, self._columns

    def copy(self):
        """Create a deep copy of the dataframe"""
        new_df = DataFrame()
        new_df.set_columns_name(self.get_column_names())
        new_df.merge(self)
        return new_df

    def head(self, n: int = 5):
        string = ""
        for column in self._data:
            string += f"{column.name}\t"
        string += "\n"
        for i in range(n):
            for column in self._data:
                string += f"{column.data[i]}\t"
            string += "\n"
        print(string)

    def print(self):
        string = ""
        for column in self._data:
            string += f"{column.name}\t"
        string += "\n"
        for i in range(len(self._data[0])):
            for column in self._data:
                string += f"{column.data[i]}\t"
            string += "\n"
        print(string)

    def add_row(self, row: list):
        if self._rows == 0 and len(self._data) == 0:  # if non-initialized dataframe,adapts its shape based on the row
            self._columns = len(row)
            for i in range(self._columns):
                new_column = Column(f"Column{i + 1}")
                self._data.append(new_column)

        if len(row) != self._columns:
            raise ValueError("Invalid row length")
        for i in range(self._columns):
            self._data[i].data.append(row[i])
        self._rows += 1

    def add_column(self, column: Column):
        if len(column.data) != self._rows and self._rows != 0:
            raise ValueError("Invalid column length")
        self._data.append(column)
        self._columns += 1

    def remove_row(self, index: int | list[int]):
        # Removing multiple rows at once it was tricky of tricky since data shrinks in the iteration and indexes change
        if isinstance(index, int):
            if index >= self._rows:
                raise ValueError("Index out of range")
            for column in self._data:
                del column.data[index]
            self._rows -= 1
        elif isinstance(index, list):
            if len(index) > self._rows:
                raise ValueError("Too many indices for this dataframe")
            sorted_list = sorted(index)
            delta = 0
            for i in sorted_list:
                if i >= self._rows:
                    raise ValueError("Index out of range")
                for column in self._data:
                    del column.data[i - delta]
                delta += 1
                self._rows -= 1

    def shuffle(self, seed: int = 1):
        random.seed(seed)
        indices = list(range(self._rows))
        random.shuffle(indices)
        temp_data = DataFrame()

        # Initialize columns in temp_data with empty data
        for column in self._data:
            new_column = Column(column.name)
            new_column.type = column.type
            new_column.type = column.type
            new_column.data = []
            temp_data._data.append(new_column)
            temp_data._columns += 1

        # Add rows in shuffled order to temp_data
        for i in indices:
            for j in range(self._columns):
                temp_data._data[j].data.append(self._data[j].data[i])
            temp_data._rows += 1

        # Replace self with temp_data
        self._data = temp_data._data
        self._rows = temp_data._rows
        self._columns = temp_data._columns

    def count_occurrences(self, value_to_count):
        count = 0
        for column in self._data:
            count += column.data.count(value_to_count)
        return count

    def remove_column(self, name: str | int):
        if isinstance(name, int):
            name = self._data[name].name
        for column in self._data:
            if column.name == name:
                self._data.remove(column)
                self._columns -= 1
                return
        raise ValueError("Column not found")

    def get_column_names(self) -> list[str]:
        return [column.name for column in self._data]

    def set_columns_name(self, name_list: list[str]):
        if len(self._data) == 0 and len(name_list) > 0:  # for a non-initialized dataframe
            # Initialize columns with empty data
            for name in name_list:
                new_column = Column(name)
                self._data.append(new_column)
            self._columns = len(name_list)
        elif len(name_list) != self._columns:
            raise ValueError("Invalid name list length")

        for i, name in enumerate(name_list):    # Set the column names
            self._data[i].name = name

    def get_column(self, value: int | str) -> Column:
        if isinstance(value, int):
            if value >= self._columns:
                raise ValueError("Column index out of range")
            return self._data[value]
        elif isinstance(value, str):
            if value not in self.get_column_names():
                raise ValueError("Column not found")
            for column in self._data:
                if column.name == value:
                    return column

    def pop_column(self, value: int | str) -> list:
        """Returns the column as a list and removes it from the dataframe
        Args:
            value (int | str): The column name or index
        Returns:
            list: The column as a list
        Raises:
            ValueError: Column not found
            ValueError: Column index out of range
            """
        column_to_return = None
        if isinstance(value, int):
            if value >= self._columns:
                raise ValueError("Column index out of range")
            column_to_return = self._data[value]
        elif isinstance(value, str):
            if value not in self.get_column_names():
                raise ValueError(f"Column {value} not found")
            for column in self._data:
                if column.name == value:
                    column_to_return = column
                    break
        self.remove_column(value)
        return column_to_return

    def pop_row(self, index: int) -> list:
        """ Returns the row as a list and removes it from the dataframe"""
        if index >= self._rows:
            raise ValueError("Row index out of range")
        row_to_return = []
        for column in self._data:
            row_to_return.append(column.data[index])
        self.remove_row(index)
        return row_to_return

    def merge(self, other: 'DataFrame', horizontal: bool = True):
        """Merge another dataframe with this dataframe, either horizontally by rows or vertically by columns"""
        if horizontal:
            for i in range(other._rows):
                self.add_row(other[i])
        else:
            for i in range(other._columns):
                self.add_column(other.get_column(i))

    def to_csv(self, filename: str):
        with open(filename, 'w', encoding='utf-8', newline='\n') as f:
            # write the column headers
            header = [column.name for column in self._data]
            f.write(','.join(header) + '\n')
            # write the rows
            for i in range(self._rows):
                row = [str(column.data[i]) for column in self._data]
                f.write(','.join(row) + '\n')

    @classmethod
    def from_csv(cls, filename: str, sep: str = ",", header: bool = True):
        new_instance = cls()
        with open(filename, "r") as file:
            if header:
                headers = file.readline().strip().split(sep)
                new_instance._data = [Column(name) for name in headers]

            for line in file:
                row = line.strip().split(sep)
                for col_index, item in enumerate(row):
                    new_instance._data[col_index].data.append(item)

        new_instance._rows = len(new_instance._data[0].data)
        new_instance._columns = len(new_instance._data)
        return new_instance

    def __getitem__(self, indices):
        if isinstance(indices, int):
            # Singleton index is for accessing a row, return a Row proxy that can be indexed again for column access
            # in double indexing [][]. I really want to implement all the possible slicing operations, but this implies
            # modifying the underlying data structure for every operation in this and the proxy classes Column and Row
            if indices >= self._rows:
                raise IndexError("Row index out of range")
            return Row(self, indices)
        elif isinstance(indices, str):
            # Direct access to columns by name
            return self.column(indices)
        elif isinstance(indices, tuple):
            # Direct cell access via (row, col)
            row, col = indices
            if row >= self._rows or col >= self._columns:
                raise IndexError("Index out of range")
            return self._data[col].data[row]

    def __setitem__(self, key: int, value):
        """Only a Column object can be assigned by indexing obj[], this is bypassed by double-indexing [][] by
        overloaded__setitem__ in Column class"""
        if isinstance(key, int):
            if not isinstance(value, Column):
                raise ValueError("Value must be a Column object")
            print("Overloaded __setitem__")
            print(self._data[key])
            self._data[key] = value

    def __str__(self):
        return f"<DataFrame with {self._rows} rows and {self._columns} columns>"

    def __iter__(self) -> iter:
        # I have kinda simulated the iterrows() from pandas here, but overloaded the iterator instead
        cnt = 0
        for i in range(self._rows):
            cnt += 1
            yield cnt - 1, self[i]
