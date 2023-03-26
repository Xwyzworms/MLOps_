const Schema = require("./employees_pb");
const fs = require("fs");

const rose = new Schema.Employee();
rose.setSalary(2990);
rose.setName("Rose");
rose.setId(101);

const tous = new Schema.Employee();
tous.setSalary(5921);
tous.setName("tous");
tous.setId(291);

const Les = new Schema.Employee();
Les.setSalary(4990);
Les.setName("Memes");
Les.setId(102);

const employees = new Schema.Employees();
employees.addEmployees(rose);
employees.addEmployees(tous);
employees.addEmployees(Les);

console.log(rose.toString());

// Preparing to send it over the network
const bytes = employees.serializeBinary();

console.log(" Binary bytes : " + bytes);

fs.writeFileSync("employeesBinary",bytes);

// Read the deserialzie data

const deserializeData = Schema.Employees.deserializeBinary(bytes);

console.log(" deserialized Data " + deserializeData);

for( let i  = 0 ; i < deserializeData.getEmployeesList().length; i++) 
{
    let contentData = new Schema.Employee();
    const employee =  deserializeData.getEmployeesList()[i];
    contentData.setId(employee["array"][0]);
    contentData.setName(employee["array"][1]);
    contentData.setSalary(employee["array"][3]);

    console.log("Content data Fr : " + contentData);
}

